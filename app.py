# app.py

import streamlit as st
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import openai
# from openai.error import RateLimitError

# ─────────── Load artifacts & data ───────────────────────────────
FEATURES_PATH   = "artifacts/features.joblib"
METRICS_PATH    = "artifacts/metrics.json"
REAL_DATA_PATH  = "data/cleaned_real.csv"
SYNTH_DATA_PATH = "data/synthesized.csv"

features = joblib.load(FEATURES_PATH)
metrics  = json.load(open(METRICS_PATH))
df_real  = pd.read_csv(REAL_DATA_PATH)
df_synth = pd.read_csv(SYNTH_DATA_PATH)

# ─────────── Configure OpenAI API key ────────────────────────────
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ─────────── Page & Sidebar Setup ─────────────────────────────────
st.set_page_config(
    page_title="GEN AI Heart Disease Prediction and Diagnosis",
    layout="wide"
)

# Sidebar controls
st.sidebar.title("Controls & Diagnostics")
model_choice = st.sidebar.selectbox(
    "Model",
    ["baseline", "augmented"],
    help="Baseline = no CTGAN augmentation; Augmented = with CTGAN"
)
if st.sidebar.button("Test OpenAI Connection"):
    try:
        resp = openai.Model.list()
        st.sidebar.success(f"Connected: {len(resp['data'])} models available")
    except Exception as e:
        st.sidebar.error(f"Connection failed: {e}")
show_llm    = st.sidebar.checkbox("Enable LLM summary", value=True)
show_tables = st.sidebar.checkbox("Show raw tables", value=False)

st.title("HEART DISEASE PREDICTION & RECOMMENDATION USING GEN AI")

# ─────────── Main Tabs ─────────────────────────────────────────────
tab_pred, tab_eda, tab_genai = st.tabs(
    ["🩺 Prediction","🔍 EDA", "🧬 Gen-AI Diagnostics"]
)



# ─────────── Tab: Prediction & Recommendation ──────────────────────
with tab_pred:
    st.header("Prediction & Recommendation")

    # Display metrics side by side
    m1, m2 = st.columns(2)
    m1.metric(
        label="Baseline Accuracy",
        value=f"{metrics['baseline']['accuracy']:.2%}",
        delta=f"AUC {metrics['baseline']['auc']:.2%}"
    )
    m2.metric(
        label="Augmented Accuracy",
        value=f"{metrics['augmented']['accuracy']:.2%}",
        delta=f"AUC {metrics['augmented']['auc']:.2%}"
    )

    st.markdown("---")
    st.subheader("Enter Patient Features")

    # Form with two columns
    with st.form("predict_form"):
        col_a, col_b = st.columns(2, gap="medium")
        inputs = {}
        for idx, feat in enumerate(features):
            pane = col_a if idx % 2 == 0 else col_b
            label = feat.upper()
            if feat in ["sex","cp","fbs","restecg","exang","slope","ca","thal"]:
                opts = sorted(df_real[feat].unique().astype(int))
                inputs[feat] = pane.selectbox(
                    label=label,
                    options=opts,
                    help=f"Categorical: {feat}"
                )
            else:
                lo, hi = float(df_real[feat].min()), float(df_real[feat].max())
                inputs[feat] = pane.slider(
                    label=label,
                    min_value=lo,
                    max_value=hi,
                    value=(lo + hi) / 2,
                    step=(hi - lo) / 100,
                    help=f"Continuous: {feat}"
                )

        submitted = st.form_submit_button("Run Prediction")

    if submitted:
        # Load model & advice engine
        model      = joblib.load(f"artifacts/model_{model_choice}.joblib")
        kmeans     = joblib.load("artifacts/advice_kmeans.joblib")
        advice_map = joblib.load("artifacts/advice_map.joblib")

        X_new = pd.DataFrame([inputs], columns=features)
        prob  = model.predict_proba(X_new)[0, 1]

        out1, out2 = st.columns([1, 2], gap="large")

        # Left column: risk + advice
        with out1:
            st.metric("Predicted Disease Risk", f"{prob:.2%}")

            leaf_new = model.apply(X_new)
            cluster  = kmeans.predict(leaf_new)[0]
            tag      = advice_map[cluster]
            advice   = {
                "exercise":   "🚶 Increase daily walk to 30 min/day",
                "monitoring": "🩺 Schedule lipid panel every 6 months",
                "medication": "💊 Consider statin therapy",
                "referral":   "👩‍⚕️ Refer to cardiologist"
            }[tag]

            st.markdown("**Recommendation:**")
            st.write(f"- {advice}")

        # Right column: LLM-driven summary
        if show_llm:
            with out2:
                st.subheader("Patient Education Summary")
                prompt = (
                    "Patient features:\n"
                    + "\n".join(f"- {k}: {v}" for k, v in inputs.items())
                    + f"\nRisk: {prob:.2%}\nNext step: {advice}\n\n"
                    + "Summarize these risk factors and recommended next steps "
                      "in 2–3 plain-language sentences."
                )
                try:
                    with st.spinner("Generating summary…"):
                        resp = openai.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role":"system", "content":"You are a helpful medical assistant."},
                                {"role":"user",   "content": prompt}
                            ],
                            temperature=0.7,
                            max_tokens=150,
                        )
                    summary = resp.choices[0].message.content
                    st.write(summary)
                except:
                    st.error("⚠️ LLM quota exceeded or billing not enabled.")
# ─────────── Tab: EDA ──────────────────────────────────────────────
with tab_eda:
    st.header("Exploratory Data Analysis")
    if show_tables:
        st.subheader("Raw real data sample")
        st.dataframe(df_real.head(10), use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(df_real.describe().round(2), use_container_width=True)

    st.subheader("Feature Distributions")
    num_cols = df_real.select_dtypes(include="number").columns.tolist()
    for i in range(0, len(num_cols), 2):
        col1, col2 = st.columns(2, gap="large")
        for j, col in enumerate(num_cols[i : i + 2]):
            with (col1 if j == 0 else col2):
                st.subheader(col)
                fig, ax = plt.subplots()
                ax.hist(df_real[col], bins=30)
                ax.set_xlabel(col)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# ─────────── Tab: Gen-AI Diagnostics ──────────────────────────────
with tab_genai:
    st.header("Generative AI Diagnostics")
    import scipy.stats as ss

    st.subheader("Feature KS distances")
    ks = {col: ss.ks_2samp(df_real[col], df_synth[col]).statistic
        for col in df_real.columns if col != "target"}
    st.json(ks)
    import matplotlib.pyplot as plt
    import scipy.stats as ss

    # compute KS
    ks = {col: ss.ks_2samp(df_real[col], df_synth[col]).statistic
        for col in df_real.columns if col!="target"}

    # bar plot
    fig, ax = plt.subplots(figsize=(6,3))
    names = list(ks.keys())
    values= list(ks.values())
    ax.barh(names, values)
    ax.set_xlabel("KS distance")
    ax.set_title("Feature‐by‐feature real vs. synth KS")
    st.pyplot(fig)
    if show_tables:
        st.subheader("Synthetic data sample")
        st.dataframe(df_synth.head(10), use_container_width=True)

    st.subheader("Real vs. Synthetic Distributions")
    compare_cols = ["chol", "age", "trestbps", "thalach"]
    for i in range(0, len(compare_cols), 2):
        col1, col2 = st.columns(2, gap="large")
        for j, col in enumerate(compare_cols[i : i + 2]):
            with (col1 if j == 0 else col2):
                st.subheader(col)
                fig, ax = plt.subplots()
                ax.hist(df_real[col], bins=30, alpha=0.6, label="Real")
                ax.hist(df_synth[col], bins=30, alpha=0.6, label="Synth")
                ax.set_xlabel(col)
                ax.legend()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
