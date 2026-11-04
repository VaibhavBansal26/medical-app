"""Heart-disease risk models: sklearn/XGBoost baselines and a PyTorch MLP, all as sklearn Pipelines."""
import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

from .data import CATEGORICAL, NUMERIC


def make_preprocessor() -> ColumnTransformer:
    """Imputation lives inside the pipeline so it is fitted on training folds only (no leakage)."""
    num = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=-1)),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEGORICAL)])


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """Small MLP (BatchNorm + Dropout + weight decay) with early stopping on a held-out slice."""

    def __init__(self, hidden=(64, 32), dropout=0.3, lr=3e-3, weight_decay=1e-3,
                 epochs=300, batch_size=64, patience=25, seed=0):
        self.hidden, self.dropout, self.lr, self.weight_decay = hidden, dropout, lr, weight_decay
        self.epochs, self.batch_size, self.patience, self.seed = epochs, batch_size, patience, seed

    def _build(self, d_in):
        layers, d = [], d_in
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(self.dropout)]
            d = h
        return nn.Sequential(*layers, nn.Linear(d, 1))

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        X = np.asarray(X, dtype=np.float32); y = np.asarray(y, dtype=np.float32)
        self.classes_ = np.array([0, 1])
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.15, stratify=y, random_state=self.seed)
        Xtr, ytr, Xva, yva = map(torch.from_numpy, (Xtr, ytr, Xva, yva))
        self.net_ = self._build(X.shape[1])
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        best, best_state, bad = np.inf, None, 0
        for _ in range(self.epochs):
            self.net_.train()
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(perm), self.batch_size):
                idx = perm[i:i + self.batch_size]
                if len(idx) < 2:  # BatchNorm needs >1 sample
                    continue
                opt.zero_grad()
                loss_fn(self.net_(Xtr[idx]).squeeze(1), ytr[idx]).backward()
                opt.step()
            self.net_.eval()
            with torch.no_grad():
                val = loss_fn(self.net_(Xva).squeeze(1), yva).item()
            if val < best - 1e-4:
                best, bad = val, 0
                best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net_.load_state_dict(best_state)
        self.net_.eval()
        return self

    def predict_proba(self, X):
        with torch.no_grad():
            p = torch.sigmoid(self.net_(torch.from_numpy(np.asarray(X, dtype=np.float32))).squeeze(1)).numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def candidates() -> dict:
    return {
        "logreg": Pipeline([("prep", make_preprocessor()),
                            ("clf", LogisticRegression(C=0.5, max_iter=2000))]),
        # sklearn's gradient boosting instead of XGBoost: xgboost's libomp segfaults when imported alongside torch on macOS
        "gboost": Pipeline([("prep", make_preprocessor()),
                            ("clf", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                                   l2_regularization=1.0, random_state=0))]),
        "torch_mlp": Pipeline([("prep", make_preprocessor()), ("clf", TorchMLPClassifier())]),
    }
