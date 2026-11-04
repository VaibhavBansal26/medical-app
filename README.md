# Heart health risk + hybrid intervention recommender

Demo: https://medical-app-vb.streamlit.app/  ·  **Educational only — not medical advice.**

## What it does
1. **Risk model** — predicts a heart-disease risk score from 11 clinical inputs.
   Candidates (all in one sklearn `Pipeline` with in-fold imputation): logistic regression, gradient boosting, **PyTorch MLP**.
2. **Hybrid recommender** — ranks 13 lifestyle / screening / clinical interventions:
   - **Content-based:** patient *needs* (lipids, BP, glucose, fitness, metabolic, cardiac work-up) matched to each intervention's target vector.
   - **Collaborative:** PyTorch **two-tower** model — patient tower (MLP on features + mean embedding of interventions the patient already liked) · item tower (id embedding + content features). Handles cold start (features only) and warm start (liked items).
   - **Hybrid:** `α·content + (1-α)·collaborative`, α tuned on a validation split. A safety rule always surfaces a cardiology referral for high scores.
3. **Education links** from the NIH/NLM **MedlinePlus** web service (free, no key).

## Data
Combined UCI Heart Disease data — Cleveland + Hungarian + Switzerland + VA Long Beach, **920 patients** (was 303 / Cleveland-only).
Missing values are imputed *inside* CV folds; `ca`/`thal` are dropped (53–66 % missing, need specialist imaging).
Prevalence ranges 36 %–94 % by site, so we also report **leave-one-site-out AUC**. See `reports/eda_summary.md`.

> **Caveat:** no public patient↔intervention feedback dataset exists. The collaborative model is bootstrapped on a
> *synthetic* interaction matrix (`recsys.simulate_interactions`). The app logs 👍/👎 to `data/feedback.csv` so it can be
> replaced with real signal. Ranking metrics on synthetic data show the pipeline works; they are not clinical evidence.

## Run
```bash
pip install -r requirements.txt
python -m medapp.eda          # reports/*.png
python -m medapp.train        # trains, logs to MLflow, registers models, exports artifacts/
streamlit run app.py
mlflow ui --backend-store-uri sqlite:///mlflow.db   # experiments + model registry
pytest tests
```

## MLflow
Experiment `heart-risk-recsys`: one parent run for risk-model selection (child run per model: CV, test, leave-one-site-out metrics)
and one run for the recommender (cold/warm Recall/NDCG/Hit@3 for popularity, content, collaborative, hybrid).
Registered models `heart-risk-classifier` and `intervention-recommender` get the alias `champion`; the app serves the exported copies in `artifacts/`.

## Fixes vs. the original version
- Removed train/test leakage (CTGAN augmentation was applied before splitting; metrics were not comparable).
- Advice artifacts no longer overwritten between models; arbitrary KMeans→advice mapping replaced by the recommender.
- Cached model loading, labelled inputs, disclaimer, optional LLM removed from the critical path, non-root Docker image.
