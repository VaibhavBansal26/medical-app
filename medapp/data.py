"""Load the combined UCI Heart Disease dataset (Cleveland + Hungarian + Switzerland + VA Long Beach)."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
COMBINED_PATH = DATA_DIR / "heart_combined.csv"

URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "heart-disease/processed.{site}.data")
SITES = ["cleveland", "hungarian", "switzerland", "va"]
COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
# `ca` and `thal` are dropped from the model: 66% / 53% missing (mostly absent outside Cleveland)
# and they require fluoroscopy / thallium imaging that a screening app user will not have.
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope"]
FEATURES = NUMERIC + CATEGORICAL

# Human-readable labels for the categorical codes (UCI data dictionary).
LABELS = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"},
    "fbs": {0: "Fasting glucose ≤120 mg/dl", 1: "Fasting glucose >120 mg/dl"},
    "restecg": {0: "Normal", 1: "ST-T abnormality", 2: "LV hypertrophy"},
    "exang": {0: "No", 1: "Yes"},
    "slope": {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
    "ca": {0: "0 vessels", 1: "1 vessel", 2: "2 vessels", 3: "3 vessels"},
    "thal": {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"},
}
DESCRIPTIONS = {
    "age": "Age (years)", "sex": "Sex", "cp": "Chest pain type",
    "trestbps": "Resting blood pressure (mm Hg)", "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar", "restecg": "Resting ECG",
    "thalach": "Max heart rate achieved (bpm)", "exang": "Exercise-induced angina",
    "oldpeak": "ST depression (exercise vs rest)", "slope": "Peak-exercise ST slope",
    "ca": "Major vessels coloured by fluoroscopy", "thal": "Thallium stress test",
}


def download() -> pd.DataFrame:
    frames = []
    for site in SITES:
        df = pd.read_csv(URL.format(site=site), names=COLS, na_values="?")
        df["site"] = site
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    # Zeros here are physiologically impossible and are used as "missing" by the source sites.
    df.loc[df["chol"] == 0, "chol"] = np.nan
    df.loc[df["trestbps"] == 0, "trestbps"] = np.nan
    df["target"] = (df["target"] > 0).astype(int)
    return df


def load(refresh: bool = False) -> pd.DataFrame:
    """Return the combined dataset (missing values kept as NaN; imputation happens in the model pipeline)."""
    if COMBINED_PATH.exists() and not refresh:
        return pd.read_csv(COMBINED_PATH)
    DATA_DIR.mkdir(exist_ok=True)
    df = download()
    df.to_csv(COMBINED_PATH, index=False)
    return df


def to_frame(inputs: dict) -> pd.DataFrame:
    """Single-patient frame with the float dtypes the pipelines were fitted on."""
    return pd.DataFrame([inputs])[FEATURES].astype(float)
