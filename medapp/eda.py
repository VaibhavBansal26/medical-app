"""Exploratory data analysis. Run: python -m medapp.eda  -> reports/*.png + reports/eda_summary.md"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from . import data

REPORTS = data.ROOT / "reports"


def main():
    REPORTS.mkdir(exist_ok=True)
    for old in REPORTS.glob("*_hist.png"):
        old.unlink()
    df = data.load()
    sns.set_theme(style="whitegrid")
    lines = ["# EDA summary", "", f"Rows: {len(df)}  |  prevalence: {df.target.mean():.1%}", ""]

    # 1) Missingness (drives the decision to drop ca/thal and to impute inside the CV pipeline)
    miss = df[data.COLS[:-1]].isna().mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    miss.plot.barh(ax=ax, color="#c0504d")
    ax.set_xlabel("fraction missing"); ax.set_title("Missing values by feature")
    fig.tight_layout(); fig.savefig(REPORTS / "missingness.png", dpi=120); plt.close(fig)
    lines += ["## Missingness", miss.round(3).to_frame("missing").to_markdown(), ""]

    # 2) Class balance per site (sites differ a lot -> site shift is a real validity risk)
    by_site = df.groupby("site").target.agg(["size", "mean"]).round(3)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.barplot(x=by_site.index, y=by_site["mean"], ax=ax, color="#4f81bd")
    ax.set_ylabel("disease prevalence"); ax.set_title("Prevalence by source site")
    fig.tight_layout(); fig.savefig(REPORTS / "prevalence_by_site.png", dpi=120); plt.close(fig)
    lines += ["## Prevalence by site", by_site.to_markdown(), ""]

    # 3) Numeric distributions split by target + Mann-Whitney effect
    fig, axes = plt.subplots(1, len(data.NUMERIC), figsize=(4 * len(data.NUMERIC), 3.5))
    rows = []
    for ax, col in zip(axes, data.NUMERIC):
        sns.kdeplot(data=df, x=col, hue="target", common_norm=False, fill=True, ax=ax)
        a, b = df.loc[df.target == 0, col].dropna(), df.loc[df.target == 1, col].dropna()
        u, p = stats.mannwhitneyu(a, b)
        rows.append({"feature": col, "median_healthy": a.median(), "median_disease": b.median(),
                     "AUC_univariate": round(u / (len(a) * len(b)), 3), "p": f"{p:.1e}"})
    fig.tight_layout(); fig.savefig(REPORTS / "numeric_by_target.png", dpi=120); plt.close(fig)
    lines += ["## Numeric features vs target", pd.DataFrame(rows).to_markdown(index=False), ""]

    # 4) Categorical vs target (chi-square)
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), data.CATEGORICAL):
        ct = df.groupby(col, dropna=False).target.mean()
        ax.bar([str(i) for i in ct.index], ct.values, color="#4f81bd")
        ax.set_title(f"{col}: P(disease)"); ax.set_ylim(0, 1)
        tab = pd.crosstab(df[col], df.target)
        chi2, p, *_ = stats.chi2_contingency(tab)
        n = tab.values.sum(); v = np.sqrt(chi2 / (n * (min(tab.shape) - 1)))
        rows.append({"feature": col, "cramers_v": round(v, 3), "p": f"{p:.1e}"})
    fig.tight_layout(); fig.savefig(REPORTS / "categorical_by_target.png", dpi=120); plt.close(fig)
    lines += ["## Categorical features vs target", pd.DataFrame(rows).to_markdown(index=False), ""]

    # 5) Correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[data.NUMERIC + ["sex", "fbs", "exang", "target"]].corr(), annot=True, fmt=".2f",
                cmap="coolwarm", center=0, ax=ax)
    fig.tight_layout(); fig.savefig(REPORTS / "correlation.png", dpi=120); plt.close(fig)

    (REPORTS / "eda_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
