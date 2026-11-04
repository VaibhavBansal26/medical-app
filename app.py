import csv
import json
from datetime import datetime, timezone

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from medapp import data, education, recsys
from medapp.risk import TorchMLPClassifier  # noqa: F401  (needed to unpickle the NN pipeline)

st.set_page_config(page_title="Heart Health Risk & Recommendations", page_icon="🫀", layout="wide")

FEEDBACK_PATH = data.ROOT / "data" / "feedback.csv"


@st.cache_resource
def load_artifacts():
    return (joblib.load(data.ROOT / "artifacts" / "risk_model.joblib"),
            joblib.load(data.ROOT / "artifacts" / "recommender.joblib"),
            json.loads((data.ROOT / "artifacts" / "metrics.json").read_text()))


@st.cache_data
def load_df():
    return data.load()


def log_feedback(item_id: str, liked: int, row: dict, risk: float):
    FEEDBACK_PATH.parent.mkdir(exist_ok=True)
    new = not FEEDBACK_PATH.exists()
    with open(FEEDBACK_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "item_id", "liked", "risk", *data.FEATURES])
        w.writerow([datetime.now(timezone.utc).isoformat(), item_id, liked, round(risk, 4),
                    *[row[c] for c in data.FEATURES]])


risk_model, recommender, metrics = load_artifacts()
df = load_df()

st.title("🫀 Heart Health Risk & Personalised Recommendations")
st.warning("Educational demo trained on 1980s UCI research data. **Not medical advice** — it cannot diagnose "
           "or treat anything. Talk to a clinician about your health.", icon="⚠️")
tab_pred, tab_eda, tab_models = st.tabs(["🩺 Risk & Recommendations", "🔍 EDA", "📈 Models & Tracking"])

# ───────────────────────── prediction + recommendations ─────────────────────────
with tab_pred:
    med = df[data.FEATURES].median(numeric_only=True)
    with st.form("patient"):
        c1, c2, c3 = st.columns(3)
        inputs = {}
        inputs["age"] = c1.slider("Age", 20, 90, int(med["age"]))
        inputs["sex"] = c1.selectbox("Sex", list(data.LABELS["sex"]), format_func=data.LABELS["sex"].get)
        inputs["trestbps"] = c1.slider(data.DESCRIPTIONS["trestbps"], 80, 220, int(med["trestbps"]))
        inputs["chol"] = c2.slider(data.DESCRIPTIONS["chol"], 100, 600, int(med["chol"]))
        inputs["thalach"] = c2.slider(data.DESCRIPTIONS["thalach"], 60, 210, int(med["thalach"]))
        inputs["oldpeak"] = c2.slider(data.DESCRIPTIONS["oldpeak"], 0.0, 6.5, 0.5, 0.1)
        for col, pane in zip(["cp", "fbs", "restecg", "exang", "slope"], [c3] * 5):
            inputs[col] = pane.selectbox(data.DESCRIPTIONS[col], list(data.LABELS[col]),
                                         format_func=data.LABELS[col].get)
        liked = st.multiselect("Interventions you have already tried and liked (improves personalisation)",
                               recsys.CATALOG.id.tolist(),
                               format_func=lambda i: recsys.CATALOG.set_index("id").name[i])
        go = st.form_submit_button("Run", type="primary")

    if go:
        st.session_state["result"] = (inputs, liked)

    if "result" in st.session_state:
        inputs, liked = st.session_state["result"]
        row = data.to_frame(inputs)
        prob = float(risk_model.predict_proba(row[data.FEATURES])[0, 1])
        left, right = st.columns([1, 2], gap="large")
        with left:
            st.metric("Estimated risk score", f"{prob:.0%}",
                      help="Model output for people in the UCI referral cohorts, where ~55% had disease. "
                           "It is not a population-risk probability.")
            band = "higher" if prob >= 0.66 else "moderate" if prob >= 0.33 else "lower"
            st.caption(f"Relative band: **{band}**. Champion model: `{metrics['champion']}`.")
        recs = recommender.recommend(row, prob, liked, k=5)
        with right:
            st.subheader("Recommended next steps")
            if recs.empty:
                st.success("Nothing in the catalog stands out for these inputs — keep up general healthy habits "
                           "and routine check-ups.")
            for _, r in recs.iterrows():
                with st.container(border=True):
                    a, b = st.columns([4, 1])
                    a.markdown(f"**{r['name']}** · _{r['type']}_  \n{r['why']}")
                    a.progress(float(r["score"]), text=f"match {r['score']:.0%}  "
                               f"(content {r['content']:.0%} · similar-patients {r['collaborative']:.0%})")
                    info = education.lookup(r["topic"])
                    if info:
                        a.markdown(f"[{info['title']} — MedlinePlus]({info['url']})")
                    if b.button("👍", key=f"up_{r['id']}"):
                        log_feedback(r["id"], 1, inputs, prob); b.success("Saved")
                    if b.button("👎", key=f"dn_{r['id']}"):
                        log_feedback(r["id"], 0, inputs, prob); b.info("Saved")

# ───────────────────────── EDA ─────────────────────────
with tab_eda:
    st.caption(f"Combined UCI Heart Disease data: {len(df)} patients from 4 sites.")
    st.dataframe(df.describe().round(2).T, use_container_width=True)
    for title, img in [("Missingness", "missingness"), ("Prevalence by site", "prevalence_by_site"),
                       ("Numeric features by outcome", "numeric_by_target"),
                       ("Categorical features vs outcome", "categorical_by_target"),
                       ("Correlation", "correlation")]:
        p = data.ROOT / "reports" / f"{img}.png"
        if p.exists():
            st.subheader(title); st.image(str(p))
    summary = data.ROOT / "reports" / "eda_summary.md"
    if summary.exists():
        with st.expander("Statistical tests"):
            st.markdown(summary.read_text())

# ───────────────────────── models ─────────────────────────
with tab_models:
    st.subheader("Risk models (5-fold CV on dev split, plus 20% untouched test)")
    rows = [{"model": n, "cv_auc": v["cv"]["auc"], "test_auc": v["test"]["auc"], "test_brier": v["test"]["brier"],
             "leave-one-site-out AUC": sum(v["loso_auc"].values()) / len(v["loso_auc"])}
            for n, v in metrics["risk"].items()]
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True)
    st.caption("Leave-one-site-out shows how much performance drops on an unseen hospital.")
    st.subheader("Recommender ranking quality (NDCG@3, held-out patients)")
    rm = metrics["recsys"]
    st.dataframe(pd.DataFrame({regime: {s: rm[f"{regime}/{s}/ndcg@3"] for s in
                                        ["popularity", "content", "collaborative", "hybrid"]}
                               for regime in ["cold", "warm"]}).round(3))
    st.info("Interaction data for the collaborative model is **synthetic** (simulated from need-matching plus "
            "latent adherence profiles). Thumbs up/down feedback is logged to `data/feedback.csv` so real "
            "signal can replace it.")
    st.code("mlflow ui --backend-store-uri sqlite:///mlflow.db", language="bash")
