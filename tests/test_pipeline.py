import numpy as np
import pandas as pd

from medapp import data, recsys, risk


def _df():
    return data.load()


def test_dataset_shape_and_missing_zeros():
    df = _df()
    assert len(df) == 920 and set(df.site) == set(data.SITES)
    assert (df["chol"].dropna() > 0).all() and (df["trestbps"].dropna() > 0).all()


def test_risk_pipeline_handles_missing_and_returns_probs():
    df = _df()
    pipe = risk.candidates()["logreg"].fit(df[data.FEATURES], df["target"])
    row = data.to_frame({c: np.nan for c in data.FEATURES} | {"age": 50, "sex": 1})
    p = pipe.predict_proba(row)[0, 1]
    assert 0 <= p <= 1


def test_recommender_scores_bounded_and_high_risk_gets_referral():
    df = _df().head(200)
    risk_vec = df["target"].values.astype(np.float32) * 0.8
    Y = recsys.simulate_interactions(df, risk_vec)
    assert Y.shape == (200, recsys.N_ITEMS)
    model = recsys.train_cf(recsys.user_features(df, risk_vec), Y, epochs=20)
    rec = recsys.HybridRecommender(model, alpha=0.5)
    row = data.to_frame(df.iloc[0][data.FEATURES].to_dict())
    out = rec.recommend(row, risk=0.95, k=3)
    assert "cardiology_referral" in out.id.values
    assert recsys.content_scores(recsys.patient_needs(df, risk_vec)).max() <= 1


def test_no_liked_items_repeated():
    df = _df().head(50)
    r = np.full(50, 0.5, np.float32)
    model = recsys.train_cf(recsys.user_features(df, r), recsys.simulate_interactions(df, r), epochs=5)
    out = recsys.HybridRecommender(model, 0.5).recommend(data.to_frame(df.iloc[0][data.FEATURES].to_dict()), 0.5,
                                                          liked=["walking_150"], k=13)
    assert "walking_150" not in out.id.values
