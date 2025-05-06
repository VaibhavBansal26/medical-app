# data_pipeline.py

import os
import json
import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata.single_table import SingleTableMetadata
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

# ── 0) Setup folders ─────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

# ── 1) Load & clean real data ────────────────────────────────────────────────
def load_and_clean() -> pd.DataFrame:
    """Download & preprocess the UCI Cleveland heart dataset."""
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "heart-disease/processed.cleveland.data"
    )
    cols = [
        "age","sex","cp","trestbps","chol","fbs","restecg",
        "thalach","exang","oldpeak","slope","ca","thal","target"
    ]
    df = pd.read_csv(url, names=cols)
    df.replace("?", pd.NA, inplace=True)
    df.dropna(inplace=True)
    df = df.astype(float)
    df["target"] = (df["target"] > 0).astype(int)
    df.to_csv("data/cleaned_real.csv", index=False)
    return df

# ── 2) Augment minority using SDV’s CTGAN ────────────────────────────────────
def augment_minority(df: pd.DataFrame) -> pd.DataFrame:
    """
    Train a CTGAN on the positive‐disease subset
    and synthesize the same number of new positives.
    """
    minority = df[df.target == 1]

    # Create and configure metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=minority)

    # Instantiate and train CTGAN
    synth = CTGANSynthesizer(metadata=metadata)
    synth.fit(minority)

    # Generate synthetic rows
    df_synth = synth.sample(len(minority))
    df_synth["target"] = 1

    # Save synthetic for diagnostics
    df_synth.to_csv("data/synthesized.csv", index=False)

    # Merge & shuffle
    return pd.concat([df, df_synth], ignore_index=True)\
             .sample(frac=1, random_state=42)

# ── 3) Train & cluster leaf‐index embeddings ─────────────────────────────────
def train_and_cluster(df: pd.DataFrame, name: str) -> dict:
    """
    Train XGBoost, record accuracy/AUC,
    then cluster its leaf‐index embeddings for advice.
    """
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Train classifier
    model = XGBClassifier(eval_metric="logloss")
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    # Save model
    joblib.dump(model, f"artifacts/model_{name}.joblib")

    # Extract leaf‐index embeddings
    leaf_train = model.apply(X_train)  # → (n_samples, n_trees)
    kmeans = KMeans(n_clusters=4, random_state=42).fit(leaf_train)
    joblib.dump(kmeans, "artifacts/advice_kmeans.joblib")

    # Define cluster → advice tag map
    advice_map = {
        0: "exercise",
        1: "monitoring",
        2: "medication",
        3: "referral"
    }
    joblib.dump(advice_map, "artifacts/advice_map.joblib")

    return {"accuracy": acc, "auc": auc}

# ── 4) Run pipeline ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) Load & clean
    df_real = load_and_clean()

    metrics = {}
    # 2) Baseline model
    metrics["baseline"] = train_and_cluster(df_real, "baseline")
    # 3) Augmented model
    df_aug = augment_minority(df_real)
    metrics["augmented"] = train_and_cluster(df_aug, "augmented")

    # 4) Save feature list & metrics
    feature_list = df_real.drop("target", axis=1).columns.tolist()
    joblib.dump(feature_list, "artifacts/features.joblib")
    with open("artifacts/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("✅ data_pipeline complete. Check data/ and artifacts/.")



# import os, json
# import pandas as pd
# from ctgan import CTGANSynthesizer
# from sklearn.model_selection import train_test_split
# from sklearn.cluster import KMeans
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score, roc_auc_score
# import joblib

# os.makedirs("data", exist_ok=True)
# os.makedirs("artifacts", exist_ok=True)

# def load_and_clean():
#     url = (
#       "https://archive.ics.uci.edu/ml/machine-learning-databases/"
#       "heart-disease/processed.cleveland.data"
#     )
#     cols = ["age","sex","cp","trestbps","chol","fbs","restecg",
#             "thalach","exang","oldpeak","slope","ca","thal","target"]
#     df = pd.read_csv(url, names=cols)
#     df.replace("?", pd.NA, inplace=True)
#     df.dropna(inplace=True)
#     df = df.astype(float)
#     df["target"] = (df["target"] > 0).astype(int)
#     df.to_csv("data/cleaned_real.csv", index=False)
#     return df

# def augment_minority(df):
#     minority = df[df.target == 1]
#     synth = CTGANSynthesizer()
#     discrete = ["sex","cp","fbs","restecg","exang","slope","ca","thal","target"]
#     synth.fit(minority, discrete_columns=discrete)
#     df_synth = synth.sample(len(minority))
#     df_synth["target"] = 1
#     df_synth.to_csv("data/synthesized.csv", index=False)
#     return pd.concat([df, df_synth], ignore_index=True).sample(frac=1, random_state=42)

# def train_and_cluster(df, name):
#     X = df.drop("target", axis=1)
#     y = df["target"]
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, stratify=y, random_state=42
#     )

#     model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
#     model.fit(X_train, y_train)

#     # metrics
#     preds = model.predict(X_test)
#     probs = model.predict_proba(X_test)[:,1]
#     acc = accuracy_score(y_test, preds)
#     auc = roc_auc_score(y_test, probs)
#     joblib.dump(model, f"artifacts/model_{name}.joblib")

#     # **Advice engine**: cluster leaf‐index embeddings
#     leaf_train = model.get_booster().predict(
#         X_train, pred_leaf=True
#     )  # shape (n_samples, n_trees)
#     kmeans = KMeans(n_clusters=4, random_state=42).fit(leaf_train)
#     joblib.dump(kmeans, "artifacts/advice_kmeans.joblib")

#     # map cluster → advice tag
#     advice_map = {
#       0: "exercise",
#       1: "monitoring",
#       2: "medication",
#       3: "referral"
#     }
#     joblib.dump(advice_map, "artifacts/advice_map.joblib")

#     return {"accuracy": acc, "auc": auc}

# if __name__ == "__main__":
#     df = load_and_clean()
#     # Baseline
#     metrics = {"baseline": train_and_cluster(df, "baseline")}

#     # Augmented
#     df_aug = augment_minority(df)
#     metrics["augmented"] = train_and_cluster(df_aug, "augmented")

#     joblib.dump(df.drop("target",1).columns.tolist(),
#                 "artifacts/features.joblib")
#     with open("artifacts/metrics.json","w") as f:
#         json.dump(metrics, f, indent=2)

#     print("✅ Pipeline done.")
