"""Train, evaluate, track (MLflow) and export everything. Run: python -m medapp.train

MLflow store: sqlite:///mlflow.db  (browse with `mlflow ui --backend-store-uri sqlite:///mlflow.db`)
Registered models: heart-risk-classifier, intervention-recommender (alias `champion` on the newest best version).
"""
import json

import joblib
import mlflow
import numpy as np
import torch
from mlflow import MlflowClient
from sklearn.base import clone
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from . import data, recsys, risk

ARTIFACTS = data.ROOT / "artifacts"
TRACKING_URI = f"sqlite:///{data.ROOT / 'mlflow.db'}"
SEED = 42


def _scores(y, p):
    return {"auc": roc_auc_score(y, p), "brier": brier_score_loss(y, p), "accuracy": accuracy_score(y, p >= 0.5)}


def leave_one_site_out(pipe, df):
    out = {}
    for site in df.site.unique():
        te = df.site == site
        if df.loc[te, "target"].nunique() < 2:
            continue
        m = clone(pipe).fit(df.loc[~te, data.FEATURES], df.loc[~te, "target"])
        out[site] = roc_auc_score(df.loc[te, "target"], m.predict_proba(df.loc[te, data.FEATURES])[:, 1])
    return out


def train_risk(df):
    dev, test = train_test_split(df, test_size=0.2, stratify=df["target"], random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    results = {}
    with mlflow.start_run(run_name="risk-model-selection") as parent:
        mlflow.log_params({"n_rows": len(df), "n_dev": len(dev), "n_test": len(test), "features": ",".join(data.FEATURES)})
        for name, pipe in risk.candidates().items():
            with mlflow.start_run(run_name=name, nested=True):
                oof = cross_val_predict(clone(pipe), dev[data.FEATURES], dev["target"], cv=cv, method="predict_proba")[:, 1]
                cvm = _scores(dev["target"], oof)
                loso = leave_one_site_out(pipe, dev)
                fitted = clone(pipe).fit(dev[data.FEATURES], dev["target"])
                tm = _scores(test["target"], fitted.predict_proba(test[data.FEATURES])[:, 1])
                mlflow.log_params({"model": name})
                mlflow.log_metrics({f"cv_{k}": v for k, v in cvm.items()})
                mlflow.log_metrics({f"test_{k}": v for k, v in tm.items()})
                mlflow.log_metrics({f"loso_auc_{s}": v for s, v in loso.items()})
                mlflow.log_metric("loso_auc_mean", float(np.mean(list(loso.values()))))
                results[name] = {"cv": cvm, "test": tm, "loso": loso, "pipe": pipe}
                print(f"{name:10s} cv_auc={cvm['auc']:.3f} test_auc={tm['auc']:.3f} brier={tm['brier']:.3f} "
                      f"loso_mean={np.mean(list(loso.values())):.3f}")
        champ = max(results, key=lambda n: results[n]["cv"]["auc"])
        final = clone(results[champ]["pipe"]).fit(df[data.FEATURES], df["target"])   # refit on all data for serving
        mlflow.log_param("champion", champ)
        info = mlflow.sklearn.log_model(final, name="model", registered_model_name="heart-risk-classifier",
                                       serialization_format="cloudpickle")
        _alias("heart-risk-classifier", info.registered_model_version, "champion")
    return champ, final, results, dev, test


def train_recsys(df):
    # out-of-fold risk so the recommender never sees an in-sample (overconfident) risk estimate
    champ = risk.candidates()
    pipe = champ[MODEL_NAME]
    oof = cross_val_predict(clone(pipe), df[data.FEATURES], df["target"],
                            cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
    oof = oof.astype(np.float32)
    Y = recsys.simulate_interactions(df, oof, seed=SEED)
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.2, random_state=SEED)
    tr, va = train_test_split(tr, test_size=0.2, random_state=SEED)
    X = recsys.user_features(df, oof)
    with mlflow.start_run(run_name="recsys-hybrid"):
        model = recsys.train_cf(X[tr], Y[tr], seed=SEED)
        alpha = recsys.tune_alpha(df, oof, Y, tr, va, model)
        metrics = recsys.evaluate(df, oof, Y, te, model, alpha)
        mlflow.log_params({"alpha_content": alpha, "embedding_dim": 16, "n_items": recsys.N_ITEMS,
                           "interactions": "synthetic-bootstrap", "interaction_density": round(float(Y.mean()), 3)})
        mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in metrics.items()})
        mlflow.log_dict(recsys.CATALOG.drop(columns="vec").to_dict("records"), "catalog.json")
        info = mlflow.pytorch.log_model(model, name="cf_model", registered_model_name="intervention-recommender",
                                       serialization_format="pickle")
        _alias("intervention-recommender", info.registered_model_version, "champion")
    for regime in ("cold", "warm"):
        print(f"[{regime}] ndcg@3  " + "  ".join(f"{s}={metrics[f'{regime}/{s}/ndcg@3']:.3f}"
              for s in ("popularity", "content", "collaborative", "hybrid")))
    return recsys.HybridRecommender(model, alpha, meta={"metrics": metrics}), metrics


def _alias(name, version, alias):
    MlflowClient().set_registered_model_alias(name, alias, version)


MODEL_NAME = "gboost"


def main():
    global MODEL_NAME
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("heart-risk-recsys")
    torch.manual_seed(SEED)
    df = data.load()
    champ, final, results, dev, test = train_risk(df)
    MODEL_NAME = champ
    rec, rec_metrics = train_recsys(df)
    ARTIFACTS.mkdir(exist_ok=True)
    for old in ARTIFACTS.glob("*.joblib"):
        old.unlink()
    joblib.dump(final, ARTIFACTS / "risk_model.joblib")
    joblib.dump(rec, ARTIFACTS / "recommender.joblib")
    summary = {"champion": champ,
               "risk": {n: {"cv": r["cv"], "test": r["test"], "loso_auc": r["loso"]} for n, r in results.items()},
               "recsys": rec_metrics, "alpha": rec.alpha}
    (ARTIFACTS / "metrics.json").write_text(json.dumps(summary, indent=2))
    print("champion:", champ, "| exported to artifacts/")


if __name__ == "__main__":
    main()
