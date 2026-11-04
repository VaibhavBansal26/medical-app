import csv
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from medapp import data, education, recsys, tracking
from medapp.risk import TorchMLPClassifier  # noqa: F401  (needed to unpickle the NN pipeline)

st.set_page_config(page_title="CardioCompass · Heart risk & next steps", page_icon="🫀", layout="wide")

FEEDBACK_PATH = data.ROOT / "data" / "feedback.csv"
TYPE_STYLE = {"lifestyle": ("#16a34a", "#dcfce7"), "screening": ("#2563eb", "#dbeafe"), "clinical": ("#e11d48", "#ffe4e6")}

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero {background: linear-gradient(120deg,#be123c 0%,#e11d48 55%,#fb7185 100%); color:#fff; padding:1.6rem 2rem;
       border-radius:18px; margin-bottom:1rem;}
.hero h1 {margin:0; font-size:2rem; color:#fff;}
.hero p {margin:.3rem 0 0; opacity:.92;}
.card {background:#fff; border:1px solid #e7e5e4; border-radius:14px; padding:1rem 1.2rem; margin-bottom:.7rem;
       box-shadow:0 1px 2px rgba(0,0,0,.04);}
.badge {display:inline-block; padding:.1rem .6rem; border-radius:999px; font-size:.72rem; font-weight:600; margin-left:.4rem;}
.meter {position:relative; height:14px; border-radius:999px; background:linear-gradient(90deg,#22c55e,#facc15,#ef4444);}
.meter .pin {position:absolute; top:-5px; width:6px; height:24px; background:#1c1917; border-radius:3px;}
.big {font-size:3rem; font-weight:700; line-height:1;}
.bar {height:8px; border-radius:999px; background:#e7e5e4; overflow:hidden; margin:.15rem 0 .5rem;}
.bar > div {height:100%; border-radius:999px;}
.muted {color:#78716c; font-size:.85rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    return (joblib.load(data.ROOT / "artifacts" / "risk_model.joblib"),
            joblib.load(data.ROOT / "artifacts" / "recommender.joblib"),
            json.loads((data.ROOT / "artifacts" / "metrics.json").read_text()))


@st.cache_data
def load_df():
    return data.load()


@st.cache_data(ttl=300)
def mlflow_runs():
    return tracking.runs()


@st.cache_data(ttl=300)
def mlflow_registry():
    return tracking.registry()


def log_feedback(item_id: str, liked: int, row: dict, risk: float):
    FEEDBACK_PATH.parent.mkdir(exist_ok=True)
    new = not FEEDBACK_PATH.exists() or FEEDBACK_PATH.stat().st_size == 0
    with open(FEEDBACK_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "item_id", "liked", "risk", *data.FEATURES])
        w.writerow([datetime.now(timezone.utc).isoformat(), item_id, liked, round(risk, 4), *[row[c] for c in data.FEATURES]])


def bar(frac: float, color: str) -> str:
    return f'<div class="bar"><div style="width:{max(0, min(1, frac)) * 100:.0f}%;background:{color}"></div></div>'


risk_model, recommender, metrics = load_artifacts()
df = load_df()

st.markdown("""<div class="hero"><h1>🫀 CardioCompass</h1>
<p>Heart-risk estimate + personalised next steps from a hybrid content / collaborative recommender.</p></div>""",
            unsafe_allow_html=True)
st.caption("⚠️ Educational demo trained on 1980s UCI research data. **Not medical advice** — it cannot diagnose or treat "
           "anything. Talk to a clinician about your health.")

tab_pred, tab_eda, tab_models, tab_mlflow = st.tabs(["🩺 Risk & next steps", "🔍 Data explorer", "📈 Model quality", "🧪 MLflow"])

# ───────────────────────── prediction + recommendations ─────────────────────────
with tab_pred:
    med = df[data.FEATURES].median(numeric_only=True)
    form_col, out_col = st.columns([1, 1.25], gap="large")
    with form_col:
        with st.form("patient"):
            st.markdown("#### Your details")
            a, b = st.columns(2)
            inputs = {}
            inputs["age"] = a.number_input("Age", 20, 90, int(med["age"]))
            inputs["sex"] = b.selectbox("Sex", list(data.LABELS["sex"]), format_func=data.LABELS["sex"].get)
            inputs["trestbps"] = a.number_input("Resting BP (mm Hg)", 80, 220, int(med["trestbps"]))
            inputs["chol"] = b.number_input("Cholesterol (mg/dl)", 100, 600, int(med["chol"]))
            inputs["thalach"] = a.number_input("Max heart rate (bpm)", 60, 210, int(med["thalach"]))
            inputs["oldpeak"] = b.number_input("ST depression", 0.0, 6.5, 0.5, 0.1)
            inputs["cp"] = st.selectbox(data.DESCRIPTIONS["cp"], list(data.LABELS["cp"]), index=3, format_func=data.LABELS["cp"].get)
            inputs["fbs"] = st.selectbox(data.DESCRIPTIONS["fbs"], list(data.LABELS["fbs"]), format_func=data.LABELS["fbs"].get)
            inputs["exang"] = st.selectbox(data.DESCRIPTIONS["exang"], list(data.LABELS["exang"]), format_func=data.LABELS["exang"].get)
            with st.expander("More ECG details"):
                inputs["restecg"] = st.selectbox(data.DESCRIPTIONS["restecg"], list(data.LABELS["restecg"]), format_func=data.LABELS["restecg"].get)
                inputs["slope"] = st.selectbox(data.DESCRIPTIONS["slope"], list(data.LABELS["slope"]), index=1, format_func=data.LABELS["slope"].get)
            names = recsys.CATALOG.set_index("id").name
            liked = st.multiselect("Already tried and liked (personalises results)", recsys.CATALOG.id.tolist(),
                                   format_func=names.get)
            go = st.form_submit_button("Get my results", type="primary", width="stretch")
        if go:
            st.session_state["result"] = (inputs, liked)

    with out_col:
        if "result" not in st.session_state:
            st.info("Fill in your details and press **Get my results**.")
        else:
            inputs, liked = st.session_state["result"]
            row = data.to_frame(inputs)
            prob = float(risk_model.predict_proba(row[data.FEATURES])[0, 1])
            band, color = (("Higher", "#ef4444") if prob >= 0.66 else ("Moderate", "#ca8a04") if prob >= 0.33 else ("Lower", "#16a34a"))
            st.markdown(f"""<div class="card">
              <div class="muted">Estimated risk score</div>
              <div class="big" style="color:{color}">{prob:.0%} <span style="font-size:1.1rem">{band}</span></div>
              <div class="meter" style="margin-top:.8rem"><div class="pin" style="left:calc({prob * 100:.0f}% - 3px)"></div></div>
              <div class="muted" style="margin-top:.6rem">Relative to patients referred for heart tests (≈55% had disease) —
              not a population risk. Model: <code>{metrics['champion']}</code>.</div></div>""", unsafe_allow_html=True)

            needs = recsys.patient_needs(row, np.array([prob], np.float32))[0]
            with st.expander("What drives your recommendations (your needs profile)"):
                for n, v in zip(recsys.NEEDS, needs):
                    st.markdown(f"**{n.replace('_', ' ').title()}** · {v:.0%}" + bar(v, "#e11d48"), unsafe_allow_html=True)

            recs = recommender.recommend(row, prob, liked, k=5)
            st.markdown("#### Recommended next steps")
            if recs.empty:
                st.success("Nothing stands out for these inputs — keep up healthy habits and routine check-ups.")
            for _, r in recs.iterrows():
                fg, bg = TYPE_STYLE[r["type"]]
                info = education.lookup(r["topic"])
                link = f' · <a href="{info["url"]}" target="_blank">{info["title"]} (MedlinePlus)</a>' if info else ""
                st.markdown(f"""<div class="card"><b>{r['name']}</b>
                  <span class="badge" style="color:{fg};background:{bg}">{r['type']}</span>
                  <div class="muted">{r['why']}{link}</div>
                  {bar(r['score'], fg)}
                  <div class="muted">match {r['score']:.0%} · content {r['content']:.0%} · similar patients {r['collaborative']:.0%}</div></div>""",
                            unsafe_allow_html=True)
                u, d, _ = st.columns([1, 1, 6])
                if u.button("👍", key=f"up_{r['id']}"):
                    log_feedback(r["id"], 1, inputs, prob); st.toast("Thanks — feedback saved")
                if d.button("👎", key=f"dn_{r['id']}"):
                    log_feedback(r["id"], 0, inputs, prob); st.toast("Thanks — feedback saved")

# ───────────────────────── EDA ─────────────────────────
with tab_eda:
    c1, c2, c3 = st.columns(3)
    c1.metric("Patients", len(df)); c2.metric("Disease prevalence", f"{df.target.mean():.0%}"); c3.metric("Source sites", df.site.nunique())
    for title, img, cap in [
        ("Missing data", "missingness", "`ca` and `thal` are too sparse to use."),
        ("Prevalence by hospital", "prevalence_by_site", "Large site shift → validate leave-one-site-out."),
        ("Numeric features by outcome", "numeric_by_target", ""),
        ("Categorical features vs outcome", "categorical_by_target", ""),
        ("Correlation", "correlation", "")]:
        p = data.ROOT / "reports" / f"{img}.png"
        if p.exists():
            st.subheader(title); st.image(str(p), caption=cap or None)
    with st.expander("Raw summary statistics"):
        st.dataframe(df.describe().round(2).T, width="stretch")
    s = data.ROOT / "reports" / "eda_summary.md"
    if s.exists():
        with st.expander("Statistical tests"):
            st.markdown(s.read_text())

# ───────────────────────── model quality ─────────────────────────
with tab_models:
    st.subheader("Risk models")
    rows = [{"model": n, "CV AUC": v["cv"]["auc"], "test AUC": v["test"]["auc"], "test Brier ↓": v["test"]["brier"],
             "leave-one-site-out AUC": sum(v["loso_auc"].values()) / len(v["loso_auc"])} for n, v in metrics["risk"].items()]
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
    st.bar_chart(pd.DataFrame(rows).set_index("model")[["CV AUC", "test AUC", "leave-one-site-out AUC"]], stack=False)
    st.subheader("Recommender ranking quality (NDCG@3, held-out patients)")
    rm = metrics["recsys"]
    q = pd.DataFrame({regime: {s: rm[f"{regime}/{s}/ndcg@3"] for s in ["popularity", "content", "collaborative", "hybrid"]}
                      for regime in ["cold", "warm"]})
    st.bar_chart(q, stack=False)
    st.info("Collaborative training interactions are **synthetic** (need-matching + latent adherence profiles). "
            "Thumbs up/down feedback is logged to `data/feedback.csv` so real signal can replace it.")

# ───────────────────────── MLflow ─────────────────────────
with tab_mlflow:
    st.caption(f"Live read of the MLflow tracking store (`mlflow.db`), experiment **{tracking.EXPERIMENT}**.")
    try:
        runs, reg = mlflow_runs(), mlflow_registry()
    except Exception as e:  # noqa: BLE001 - UI should degrade gracefully if the store is missing
        st.warning(f"MLflow store unavailable: {e}")
        runs, reg = pd.DataFrame(), pd.DataFrame()
    if not reg.empty:
        st.subheader("Model registry")
        st.dataframe(reg, hide_index=True, width="stretch")
    if not runs.empty:
        st.subheader("Risk-model runs")
        r = runs[runs.get("param:model").notna()] if "param:model" in runs else runs
        cols = [c for c in ["run", "start", "cv_auc", "cv_brier", "test_auc", "test_brier", "loso_auc_mean"] if c in r]
        st.dataframe(r[cols].round(3), hide_index=True, width="stretch")
        rec = runs[runs["run"] == "recsys-hybrid"]
        if not rec.empty:
            st.subheader("Recommender run")
            m = rec.iloc[0]
            k = {c: m[c] for c in rec.columns if "/" in c and c.endswith("ndcg_at_3")}
            st.dataframe(pd.Series(k, name="NDCG@3").round(3).to_frame(), width="stretch")
    st.markdown("Open the full MLflow UI locally:")
    st.code("mlflow ui --backend-store-uri sqlite:///mlflow.db", language="bash")
