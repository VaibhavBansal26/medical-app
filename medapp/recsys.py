"""Hybrid intervention recommender: content-based + collaborative (PyTorch two-tower).

Interaction data caveat: no public patient-intervention feedback dataset exists. `simulate_interactions`
bootstraps a *synthetic* implicit-feedback matrix from guideline-style need/intervention matching plus latent
patient "adherence archetypes". The Streamlit app logs real accept/reject feedback to data/feedback.csv; replace the simulator with that
so the collaborative signal gradually becomes real once enough is collected.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn

NEEDS = ["lipids", "bp", "glucose", "fitness", "metabolic", "cardiac_workup"]
TYPES = ["lifestyle", "screening", "clinical"]

# id, name, type, MedlinePlus search term, target vector over NEEDS
_ITEMS = [
    ("mediterranean_diet", "Mediterranean-style diet", "lifestyle", "Mediterranean diet", [.7, .5, .4, 0, .8, 0]),
    ("walking_150", "Brisk walking, 150 min/week", "lifestyle", "exercise and physical fitness", [.3, .4, .4, .8, .5, 0]),
    ("cardiac_rehab", "Supervised exercise / cardiac rehab", "clinical", "cardiac rehabilitation", [.2, .3, .2, 1, .2, .4]),
    ("dash_sodium", "DASH diet and sodium reduction", "lifestyle", "DASH eating plan", [.2, 1, 0, 0, .3, 0]),
    ("weight_program", "Structured weight-management program", "lifestyle", "weight control", [.3, .3, .5, .3, 1, 0]),
    ("lipid_review", "Lipid panel and treatment discussion with a clinician", "clinical", "cholesterol", [1, 0, 0, 0, .2, .1]),
    ("bp_monitoring", "Home blood-pressure monitoring and review", "screening", "high blood pressure", [0, 1, 0, 0, 0, .1]),
    ("hba1c", "HbA1c / diabetes screening", "screening", "diabetes screening", [0, .1, 1, 0, .3, 0]),
    ("stress_test", "Exercise stress test / ECG follow-up", "screening", "stress test", [0, 0, 0, .1, 0, 1]),
    ("echo", "Echocardiogram", "screening", "echocardiography", [0, .1, 0, 0, 0, .9]),
    ("cardiology_referral", "Cardiology referral", "clinical", "cardiologist", [.2, .2, 0, 0, 0, 1]),
    ("stress_sleep", "Stress management and sleep hygiene", "lifestyle", "stress", [0, .3, .1, .1, .2, 0]),
    ("Alcohol: tips for cutting back", "Reduce alcohol intake", "lifestyle", "Alcohol: tips for cutting back", [.2, .4, 0, 0, .2, 0]),
]
CATALOG = pd.DataFrame(_ITEMS, columns=["id", "name", "type", "topic", "vec"])
ITEM_VECS = np.array(CATALOG["vec"].tolist(), dtype=np.float32)          # (n_items, n_needs)
ITEM_TYPE = np.eye(len(TYPES), dtype=np.float32)[CATALOG["type"].map(TYPES.index).values]
ITEM_FEATS = np.hstack([ITEM_VECS, ITEM_TYPE])
N_ITEMS = len(CATALOG)
MIN_CONTENT, HIGH_RISK = 0.25, 0.66


# ─────────────────────────── patient representation ───────────────────────────
def patient_needs(df: pd.DataFrame, risk: np.ndarray) -> np.ndarray:
    """Map raw patient features + model risk to a 0-1 need score per target (missing -> neutral)."""
    clip = lambda x: np.clip(x, 0, 1)
    chol = df["chol"].fillna(210).values
    bp = df["trestbps"].fillna(125).values
    fbs = df["fbs"].fillna(0).values
    age = df["age"].values
    hr = df["thalach"].fillna(df["thalach"].median()).values
    exang = df["exang"].fillna(0).values
    oldpeak = df["oldpeak"].fillna(0).values
    lipids = clip((chol - 200) / 80)
    bpn = clip((bp - 120) / 40)
    glucose = np.where(fbs == 1, 1.0, 0.15)
    fitness = clip((0.85 * (220 - age) - hr) / 40 + 0.3 * exang)
    metabolic = clip(0.8 * (lipids + bpn + glucose) / 3 + 0.2 * fitness)
    workup = clip(0.5 * risk + 0.25 * clip(oldpeak / 3) + 0.25 * exang)
    return np.column_stack([lipids, bpn, glucose, fitness, metabolic, workup]).astype(np.float32)


def user_features(df: pd.DataFrame, risk: np.ndarray) -> np.ndarray:
    needs = patient_needs(df, risk)
    return np.column_stack([needs, (df["age"].values - 54) / 9, df["sex"].values, risk]).astype(np.float32)


# ─────────────────────────── content-based ───────────────────────────
def content_scores(needs: np.ndarray) -> np.ndarray:
    """Cosine-style match between patient needs and each intervention's target vector."""
    norm = np.linalg.norm(ITEM_VECS, axis=1)
    return np.clip(needs @ ITEM_VECS.T / norm, 0, 1)


# ─────────────────────────── synthetic interaction bootstrap ───────────────────────────
def simulate_interactions(df: pd.DataFrame, risk: np.ndarray, seed: int = 0) -> np.ndarray:
    """Binary (n_patients, n_items) 'accepted / adhered' matrix. SYNTHETIC; see module docstring."""
    rng = np.random.default_rng(seed)
    needs = patient_needs(df, risk)
    # latent adherence archetype: 0 lifestyle-prone, 1 clinic-prone, 2 balanced (older patients lean clinic-prone)
    p_clinic = 1 / (1 + np.exp(-(df["age"].values - 58) / 6))
    arch = np.where(rng.random(len(df)) < 0.25, 2, (rng.random(len(df)) < p_clinic).astype(int))
    bias = np.array([[1.0, -0.2, -0.9], [-0.9, 0.3, 1.0], [0.0, 0.0, 0.0]])   # archetype x type
    type_idx = CATALOG["type"].map(TYPES.index).values
    pop = rng.normal(0, 0.4, N_ITEMS)
    logit = 4.0 * content_scores(needs) - 3.4 + bias[arch][:, type_idx] + pop + rng.normal(0, 0.4, (len(df), N_ITEMS))
    return (rng.random(logit.shape) < 1 / (1 + np.exp(-logit))).astype(np.float32)


# ─────────────────────────── collaborative model ───────────────────────────
class TwoTower(nn.Module):
    """user tower: MLP(patient features) + mean embedding of items already accepted (warm start).
    item tower: learned id embedding + linear map of content features (so new items are not cold)."""

    def __init__(self, n_user_feats: int, dim: int = 16):
        super().__init__()
        self.user_mlp = nn.Sequential(nn.Linear(n_user_feats, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, dim))
        self.item_id = nn.Embedding(N_ITEMS, dim)
        self.item_content = nn.Linear(ITEM_FEATS.shape[1], dim, bias=False)
        self.item_bias = nn.Parameter(torch.zeros(N_ITEMS))
        self.register_buffer("item_feats", torch.from_numpy(ITEM_FEATS))

    def item_vecs(self):
        return self.item_id.weight + self.item_content(self.item_feats)

    def user_vec(self, x, hist):
        iv = self.item_vecs()
        pooled = hist @ iv / hist.sum(1, keepdim=True).clamp(min=1)
        return self.user_mlp(x) + pooled

    def forward(self, x, hist):
        return self.user_vec(x, hist) @ self.item_vecs().T + self.item_bias


def train_cf(X: np.ndarray, Y: np.ndarray, epochs=250, lr=5e-3, wd=1e-4, seed=0) -> TwoTower:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X_t, Y_t = torch.from_numpy(X), torch.from_numpy(Y)
    model = TwoTower(X.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        model.train()
        # random history = subset of known positives (half the users get none -> trains the cold path)
        keep = torch.from_numpy((rng.random(Y.shape) < 0.3) & (rng.random((len(Y), 1)) < 0.5)).float()
        hist = Y_t * keep
        logits = model(X_t, hist)
        loss = (nn.functional.binary_cross_entropy_with_logits(logits, Y_t, reduction="none") * (1 - hist)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


def cf_scores(model: TwoTower, X: np.ndarray, hist: np.ndarray | None = None) -> np.ndarray:
    hist = np.zeros((len(X), N_ITEMS), np.float32) if hist is None else hist.astype(np.float32)
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(X), torch.from_numpy(hist))).numpy()


# ─────────────────────────── hybrid + evaluation ───────────────────────────
def _minmax(s):
    lo, hi = s.min(1, keepdims=True), s.max(1, keepdims=True)
    return (s - lo) / np.maximum(hi - lo, 1e-9)


def blend(content: np.ndarray, cf: np.ndarray, alpha: float) -> np.ndarray:
    # both inputs are already on a 0-1 absolute scale; per-user min-max would inflate irrelevant items for low-need patients
    return alpha * content + (1 - alpha) * cf


def ranking_metrics(scores: np.ndarray, Y: np.ndarray, k: int = 3, exclude: np.ndarray | None = None) -> dict:
    s = scores.copy()
    if exclude is not None:
        s[exclude > 0] = -np.inf
        Y = Y * (1 - exclude)
    top = np.argsort(-s, axis=1)[:, :k]
    hits = np.take_along_axis(Y, top, 1)
    n_pos = Y.sum(1)
    valid = n_pos > 0
    disc = 1 / np.log2(np.arange(2, k + 2))
    ideal = np.array([disc[: int(min(n, k))].sum() for n in n_pos])
    return {f"recall@{k}": float((hits.sum(1)[valid] / np.minimum(n_pos, k)[valid]).mean()),
            f"ndcg@{k}": float(((hits * disc).sum(1)[valid] / ideal[valid]).mean()),
            f"hit@{k}": float((hits.sum(1)[valid] > 0).mean())}


@dataclass
class HybridRecommender:
    cf_model: TwoTower
    alpha: float
    meta: dict = field(default_factory=dict)

    def recommend(self, row: pd.DataFrame, risk: float, liked: list[str] | None = None, k: int = 5) -> pd.DataFrame:
        r = np.array([risk], np.float32)
        needs = patient_needs(row, r)
        hist = np.zeros((1, N_ITEMS), np.float32)
        for i in liked or []:
            hist[0, CATALOG.index[CATALOG.id == i][0]] = 1
        c = content_scores(needs)
        f = cf_scores(self.cf_model, user_features(row, r), hist)
        s = blend(c, f, self.alpha)[0]
        out = CATALOG[["id", "name", "type", "topic"]].copy()
        out["score"], out["content"], out["collaborative"] = s, c[0], f[0]
        top_need = np.array(NEEDS)[np.argsort(-(needs[0][None, :] * ITEM_VECS), axis=1)[:, :2]]
        out["why"] = [f"targets your {', '.join(t)}" for t in top_need]
        out = out[~out.id.isin(liked or [])].sort_values("score", ascending=False)
        # items with almost no matching need are noise: show nothing rather than over-medicalise a low-risk patient
        out = out[out.content >= MIN_CONTENT]
        if risk >= HIGH_RISK and "cardiology_referral" not in out.head(k).id.values:
            # clinical safety rule: a high score always surfaces clinician review regardless of learned preferences
            pin = CATALOG[CATALOG.id == "cardiology_referral"][["id", "name", "type", "topic"]].assign(
                score=1.0, content=1.0, collaborative=1.0, why="safety rule: high score warrants clinician review")
            out = pd.concat([pin, out[out.id != "cardiology_referral"]])
        return out.head(k).reset_index(drop=True)


def evaluate(df, risk, Y, test_idx, model, alpha, k=3, seed=0) -> dict:
    """Cold: patient features only. Warm: one accepted item revealed, rest held out. Returns metric dict per scorer."""
    rng = np.random.default_rng(seed)
    X = user_features(df, risk)[test_idx]
    Yt = Y[test_idx]
    content = content_scores(patient_needs(df, risk)[test_idx])
    pop = np.tile(Y.mean(0), (len(test_idx), 1))
    hist = np.zeros_like(Yt)
    for i, row in enumerate(Yt):
        pos = np.flatnonzero(row)
        if len(pos) > 1:
            hist[i, rng.choice(pos)] = 1
    res = {}
    for regime, h in [("cold", None), ("warm", hist)]:
        cf = cf_scores(model, X, h)
        scorers = {"popularity": pop, "content": content, "collaborative": cf, "hybrid": blend(content, cf, alpha)}
        for name, s in scorers.items():
            for m, v in ranking_metrics(s, Yt, k, exclude=h).items():
                res[f"{regime}/{name}/{m}"] = v
    return res


def tune_alpha(df, risk, Y, tr_idx, val_idx, model) -> float:
    X = user_features(df, risk)
    c = content_scores(patient_needs(df, risk))
    f = cf_scores(model, X[val_idx])
    grid = np.linspace(0, 1, 11)
    scores = [ranking_metrics(blend(c[val_idx], f, a), Y[val_idx])["ndcg@3"] for a in grid]
    return float(grid[int(np.argmax(scores))])
