"""
Software defect prediction experiment script.
Full combinatorial comparison: feature selection × class imbalance handling × classifier.
"""

import subprocess, sys

REQUIRED = ["scipy", "scikit-learn", "imbalanced-learn", "xgboost", "pandas", "openpyxl", "numpy"]
for pkg in REQUIRED:
    try:
        __import__(pkg.replace("-", "_").split("==")[0])
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

import os, warnings
import numpy as np
import pandas as pd
from scipy.io import arff

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────
DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR   = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)

# Skip empty/invalid datasets
SKIP = {"KC4"}

# ── Classifiers ───────────────────────────────────────
CLASSIFIERS = {
    "LR":  LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    "RF":  RandomForestClassifier(n_estimators=100, random_state=42),
    "XGB": XGBClassifier(n_estimators=100, use_label_encoder=False,
                         eval_metric="logloss", random_state=42, verbosity=0),
}

# ── Feature selection (top 50% of features) ───────────
def make_fs(name, k):
    if name == "None":
        return None
    if name == "IG":
        return SelectKBest(mutual_info_classif, k=k)
    if name == "CFS":
        return SelectKBest(f_classif, k=k)

FS_METHODS = ["None", "IG", "CFS"]

# ── Imbalance handling ────────────────────────────────
IMBALANCE = {
    "None":  None,
    "SMOTE": SMOTE(random_state=42, k_neighbors=3),
    "ADASYN": ADASYN(random_state=42, n_neighbors=3),
}

# ── Evaluation metrics ────────────────────────────────
SCORING = {
    "precision": make_scorer(precision_score, zero_division=0),
    "recall":    make_scorer(recall_score,    zero_division=0),
    "f1":        make_scorer(f1_score,        zero_division=0),
    "auc":       make_scorer(roc_auc_score,   response_method="predict_proba"),
}

CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


def load_arff(path):
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    # Decode byte columns
    for col in df.select_dtypes([object]):
        df[col] = df[col].str.decode("utf-8")
    # Target column (last column, typically "Defective")
    target_col = df.columns[-1]
    df[target_col] = (df[target_col].str.upper() == "Y").astype(int)
    df = df.dropna()
    X = df.drop(columns=[target_col]).astype(float)
    y = df[target_col]
    return X, y


def run_one(X, y, fs_name, imb_name, clf_name, clf):
    k = max(1, X.shape[1] // 2)
    fs = make_fs(fs_name, k)
    sampler = IMBALANCE[imb_name]

    steps = [("scaler", StandardScaler())]
    if fs:
        steps.append(("fs", fs))
    if sampler:
        steps.append(("sampler", sampler))
    steps.append(("clf", clf))

    pipe = ImbPipeline(steps)
    scores = cross_validate(pipe, X, y, cv=CV, scoring=SCORING,
                            error_score="raise", n_jobs=-1)
    return {
        "precision": np.nanmean(scores["test_precision"]),
        "recall":    np.nanmean(scores["test_recall"]),
        "f1":        np.nanmean(scores["test_f1"]),
        "auc":       np.nanmean(scores["test_auc"]),
    }


def main():
    records = []
    arff_files = [f for f in os.listdir(DATA_DIR)
                  if f.endswith(".arff") and os.path.splitext(f)[0] not in SKIP
                  and os.path.getsize(os.path.join(DATA_DIR, f)) > 200]

    total = len(arff_files) * len(FS_METHODS) * len(IMBALANCE) * len(CLASSIFIERS)
    done = 0

    for fname in sorted(arff_files):
        ds_name = os.path.splitext(fname)[0]
        path = os.path.join(DATA_DIR, fname)
        print(f"\n{'='*55}")
        print(f"Dataset: {ds_name}")

        try:
            X, y = load_arff(path)
        except Exception as e:
            print(f"  Load failed: {e}")
            continue

        defect_rate = y.mean() * 100
        print(f"Samples: {len(y)}  Features: {X.shape[1]}  Defect rate: {defect_rate:.1f}%")

        # Guard against ADASYN failure when the minority class is very small
        min_class = y.value_counts().min()

        for fs_name in FS_METHODS:
            for imb_name in list(IMBALANCE.keys()):
                # Skip ADASYN when minority class is too small
                if imb_name == "ADASYN" and min_class < 10:
                    done += len(CLASSIFIERS)
                    continue

                for clf_name, clf in CLASSIFIERS.items():
                    tag = f"  [{fs_name}+{imb_name}+{clf_name}]"
                    try:
                        m = run_one(X, y, fs_name, imb_name, clf_name, clf)
                        records.append({
                            "Dataset": ds_name,
                            "FS": fs_name,
                            "Imbalance": imb_name,
                            "Classifier": clf_name,
                            "Precision": round(float(m["precision"]), 4),
                            "Recall":    round(float(m["recall"]),    4),
                            "F1":        round(float(m["f1"]),        4),
                            "AUC":       round(float(m["auc"]),       4) if not np.isnan(m["auc"]) else None,
                        })
                        done += 1
                        print(f"{tag}  F1={m['f1']:.3f}  AUC={m['auc']:.3f}"
                              f"  [{done}/{total}]")
                    except Exception as e:
                        done += 1
                        print(f"{tag}  SKIP ({e})")

    # ── Save results ──────────────────────────────────
    df_all = pd.DataFrame(records)
    csv_path  = os.path.join(OUT_DIR, "all_results.csv")
    xlsx_path = os.path.join(OUT_DIR, "all_results.xlsx")
    df_all.to_csv(csv_path, index=False)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_all.to_excel(writer, sheet_name="All", index=False)

        # One sheet per dataset
        for ds in df_all["Dataset"].unique():
            df_all[df_all["Dataset"] == ds].to_excel(
                writer, sheet_name=ds[:31], index=False)

        # Best combination summary
        best = (df_all.sort_values("F1", ascending=False)
                      .groupby("Dataset").first().reset_index())
        best.to_excel(writer, sheet_name="BestPerDataset", index=False)

    print(f"\n{'='*55}")
    print(f"Experiment complete. Results saved:")
    print(f"  CSV  → {csv_path}")
    print(f"  Excel→ {xlsx_path}")
    print(f"Total records: {len(df_all)}")


if __name__ == "__main__":
    main()
