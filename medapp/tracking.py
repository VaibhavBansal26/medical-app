"""Read-only views of the MLflow tracking store + model registry for the Streamlit app."""
import mlflow
import pandas as pd
from mlflow import MlflowClient

from .data import ROOT

TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
EXPERIMENT = "heart-risk-recsys"


def _client() -> MlflowClient:
    mlflow.set_tracking_uri(TRACKING_URI)
    return MlflowClient()


def runs() -> pd.DataFrame:
    """One row per run: name, parent, start time, params and metrics (flattened)."""
    mlflow.set_tracking_uri(TRACKING_URI)
    df = mlflow.search_runs(experiment_names=[EXPERIMENT], output_format="pandas")
    if df.empty:
        return df
    df = df.rename(columns=lambda c: c.replace("metrics.", "").replace("params.", "param:")
                                       .replace("tags.mlflow.runName", "run").replace("tags.mlflow.parentRunId", "parent"))
    df["start"] = pd.to_datetime(df["start_time"]).dt.strftime("%Y-%m-%d %H:%M")
    return df


def registry() -> pd.DataFrame:
    c = _client()
    rows = []
    for m in c.search_registered_models():
        for v in c.search_model_versions(f"name='{m.name}'"):
            aliases = [a for a, ver in m.aliases.items() if str(ver) == str(v.version)]
            rows.append({"model": m.name, "version": int(v.version), "aliases": ", ".join(aliases) or "—",
                         "status": v.status, "run_id": v.run_id[:8],
                         "created": pd.to_datetime(v.creation_timestamp, unit="ms").strftime("%Y-%m-%d %H:%M")})
    return pd.DataFrame(rows)
