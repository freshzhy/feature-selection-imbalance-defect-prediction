"""
Figure generation script for the manuscript.
saves them to the figures/ directory.
"""

import subprocess, sys
for pkg in ["matplotlib", "seaborn", "pandas", "numpy", "scipy"]:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

# ── Basic settings ────────────────────────────────────────
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

PALETTE = {
    "None":  "#95a5a6",
    "SMOTE": "#3498db",
    "ADASYN":"#e74c3c",
}
CLF_COLORS = ["#2ecc71", "#3498db", "#e67e22", "#9b59b6"]
CLF_ORDER  = ["LR", "SVM", "RF", "XGB"]
FS_ORDER   = ["None", "IG", "CFS"]
IMB_ORDER  = ["None", "SMOTE", "ADASYN"]

# ── Load data ─────────────────────────────────────────────
CSV = os.path.join(os.path.dirname(__file__), "results", "all_results.csv")
df  = pd.read_csv(CSV)
df["FS"]        = df["FS"].fillna("None")
df["Imbalance"] = df["Imbalance"].fillna("None")

DATASETS = sorted(df["Dataset"].unique())

def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {name}")

# ════════════════════════════════════════════════════════
# Figure 2 — Average F1 by classifier × imbalance handling method (mean across all datasets)
# ════════════════════════════════════════════════════════
def fig2_clf_imb_f1():
    agg = df.groupby(["Classifier", "Imbalance"])["F1"].mean().reset_index()
    agg["Classifier"] = pd.Categorical(agg["Classifier"], CLF_ORDER)
    agg["Imbalance"]  = pd.Categorical(agg["Imbalance"],  IMB_ORDER)
    agg = agg.sort_values(["Classifier", "Imbalance"])

    fig, ax = plt.subplots(figsize=(9, 5))
    x      = np.arange(len(CLF_ORDER))
    width  = 0.25
    for i, imb in enumerate(IMB_ORDER):
        vals = [agg.loc[(agg.Classifier == c) & (agg.Imbalance == imb), "F1"].values[0]
                for c in CLF_ORDER]
        bars = ax.bar(x + (i - 1) * width, vals, width,
                      label=imb, color=PALETTE[imb], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(CLF_ORDER)
    ax.set_xlabel("Classifier")
    ax.set_ylabel("Average F1-score")
    ax.legend(title="Imbalance Handling", loc="upper right")
    ax.set_ylim(0, 0.65)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    return fig

# ════════════════════════════════════════════════════════
# Figure 1 — Average F1 comparison by feature selection method
# ════════════════════════════════════════════════════════
def fig1_fs_effect():
    agg = df.groupby(["FS", "Classifier"])["F1"].mean().reset_index()
    agg["FS"]         = pd.Categorical(agg["FS"],         FS_ORDER)
    agg["Classifier"] = pd.Categorical(agg["Classifier"], CLF_ORDER)
    agg = agg.sort_values(["FS", "Classifier"])

    fig, ax = plt.subplots(figsize=(9, 5))
    x     = np.arange(len(FS_ORDER))
    width = 0.2
    for i, clf in enumerate(CLF_ORDER):
        vals = [agg.loc[(agg.FS == fs) & (agg.Classifier == clf), "F1"].values[0]
                for fs in FS_ORDER]
        bars = ax.bar(x + (i - 1.5) * width, vals, width,
                      label=clf, color=CLF_COLORS[i], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(["No FS", "Information Gain", "ANOVA F-test"])
    ax.set_xlabel("Feature Selection Method")
    ax.set_ylabel("Average F1-score")
    ax.legend(title="Classifier", loc="upper right")
    ax.set_ylim(0, 0.62)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    return fig

# ════════════════════════════════════════════════════════
# Figure 3 — Heatmap: best F1 per dataset (max over classifiers; mean over FS × Imbalance)
# ════════════════════════════════════════════════════════
def fig3_heatmap_dataset_clf():
    pivot = df.groupby(["Dataset", "Classifier"])["F1"].max().reset_index()
    pivot = pivot.pivot(index="Dataset", columns="Classifier", values="F1")
    pivot = pivot[CLF_ORDER]

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd",
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Best F1-score"},
                ax=ax, vmin=0.1, vmax=0.75)
    ax.set_xlabel("Classifier")
    ax.set_ylabel("Dataset")
    return fig

# ════════════════════════════════════════════════════════
# Figure 4 — Heatmap: average F1 by imbalance handling × feature selection (all datasets)
# ════════════════════════════════════════════════════════
def fig4_heatmap_fs_imb():
    pivot = df.groupby(["FS", "Imbalance"])["F1"].mean().reset_index()
    pivot = pivot.pivot(index="FS", columns="Imbalance", values="F1")
    pivot = pivot.loc[FS_ORDER, IMB_ORDER]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Blues",
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Average F1-score"},
                ax=ax, vmin=0.2, vmax=0.55)
    ax.set_xlabel("Imbalance Handling Method")
    ax.set_ylabel("Feature Selection Method")
    ax.set_yticklabels(["No FS", "Information Gain", "ANOVA F-test"], rotation=0)
    return fig

# ════════════════════════════════════════════════════════
# Figure 5 — F1 improvement comparison: SMOTE/ADASYN vs. no handling, per dataset
# ════════════════════════════════════════════════════════
def fig5_improvement():
    base = df[df.Imbalance == "None"].groupby("Dataset")["F1"].mean()
    smote= df[df.Imbalance == "SMOTE"].groupby("Dataset")["F1"].mean()
    adas = df[df.Imbalance == "ADASYN"].groupby("Dataset")["F1"].mean()

    datasets = sorted(base.index)
    x = np.arange(len(datasets))
    width = 0.3

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, [base[d]  for d in datasets], width, label="No Handling",
           color=PALETTE["None"],  edgecolor="white")
    ax.bar(x,         [smote[d] for d in datasets], width, label="SMOTE",
           color=PALETTE["SMOTE"], edgecolor="white")
    ax.bar(x + width, [adas[d]  for d in datasets], width, label="ADASYN",
           color=PALETTE["ADASYN"],edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Average F1-score")
    ax.legend(title="Imbalance Handling")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    return fig

# ════════════════════════════════════════════════════════
# Figure 6 — AUC heatmap: dataset × classifier (best AUC)
# ════════════════════════════════════════════════════════
def fig6_auc_heatmap():
    pivot = df.groupby(["Dataset", "Classifier"])["AUC"].max().reset_index()
    pivot = pivot.pivot(index="Dataset", columns="Classifier", values="AUC")
    pivot = pivot[CLF_ORDER]

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="PuBu",
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Best AUC-ROC"},
                ax=ax, vmin=0.5, vmax=1.0)
    ax.set_xlabel("Classifier")
    ax.set_ylabel("Dataset")
    return fig

# ════════════════════════════════════════════════════════
# Figure 7 — Radar chart: overall performance of each strategy across four metrics
# ════════════════════════════════════════════════════════
def fig7_radar():
    metrics = ["F1", "AUC", "Precision", "Recall"]
    combos  = {
        "No FS + No Handling": df[(df.FS == "None") & (df.Imbalance == "None")],
        "IG + SMOTE":          df[(df.FS == "IG")   & (df.Imbalance == "SMOTE")],
        "AF + SMOTE":          df[(df.FS == "CFS")  & (df.Imbalance == "SMOTE")],
        "IG + ADASYN":         df[(df.FS == "IG")   & (df.Imbalance == "ADASYN")],
        "AF + ADASYN":         df[(df.FS == "CFS")  & (df.Imbalance == "ADASYN")],
    }

    values = {k: [v[m].mean() for m in metrics] for k, v in combos.items()}

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    colors = ["#95a5a6", "#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})

    for (label, vals), color in zip(values.items(), colors):
        v = vals + vals[:1]
        ax.plot(angles, v, "o-", linewidth=2, color=color, label=label)
        ax.fill(angles, v, alpha=0.07, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, size=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], size=8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    return fig

# ════════════════════════════════════════════════════════
# ── Main ──────────────────────────────────────────────────
# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating manuscript figures...")
    save(fig2_clf_imb_f1(),    "fig2_classifier_imbalance_f1.png")
    save(fig1_fs_effect(),     "fig1_feature_selection_effect.png")
    save(fig5_heatmap_dataset_clf(), "fig5_heatmap_dataset_classifier_f1.png")
    save(fig4_heatmap_fs_imb(),"fig4_heatmap_fs_imbalance.png")
    save(fig3_improvement(),   "fig3_imbalance_improvement_by_dataset.png")
    save(fig6_auc_heatmap(),   "fig6_heatmap_auc.png")
    save(fig7_radar(),         "fig7_radar_strategy_comparison.png")

    print(f"\nAll figures saved to: figures/")
