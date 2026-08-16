"""
软件缺陷预测实验脚本
特征选择 × 类别不平衡处理 × 分类器 全组合对比实验
"""

import subprocess, sys

REQUIRED = ["scipy", "scikit-learn", "imbalanced-learn", "xgboost", "pandas", "openpyxl", "numpy"]
for pkg in REQUIRED:
    try:
        __import__(pkg.replace("-", "_").split("==")[0])
    except ImportError:
        print(f"安装 {pkg}...")
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

# ── 配置 ──────────────────────────────────────────────
DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR   = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)

# 跳过空文件
SKIP = {"KC4"}

# ── 分类器 ────────────────────────────────────────────
CLASSIFIERS = {
    "LR":  LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    "RF":  RandomForestClassifier(n_estimators=100, random_state=42),
    "XGB": XGBClassifier(n_estimators=100, use_label_encoder=False,
                         eval_metric="logloss", random_state=42, verbosity=0),
}

# ── 特征选择 (选择前 50% 特征) ────────────────────────
def make_fs(name, k):
    if name == "None":
        return None
    if name == "IG":
        return SelectKBest(mutual_info_classif, k=k)
    if name == "CFS":
        return SelectKBest(f_classif, k=k)

FS_METHODS = ["None", "IG", "CFS"]

# ── 不平衡处理 ────────────────────────────────────────
IMBALANCE = {
    "None":  None,
    "SMOTE": SMOTE(random_state=42, k_neighbors=3),
    "ADASYN": ADASYN(random_state=42, n_neighbors=3),
}

# ── 评估指标 ──────────────────────────────────────────
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
    # 解码字节列
    for col in df.select_dtypes([object]):
        df[col] = df[col].str.decode("utf-8")
    # 目标列（最后一列，通常是 Defective）
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
        print(f"数据集: {ds_name}")

        try:
            X, y = load_arff(path)
        except Exception as e:
            print(f"  加载失败: {e}")
            continue

        defect_rate = y.mean() * 100
        print(f"样本: {len(y)}  特征: {X.shape[1]}  缺陷率: {defect_rate:.1f}%")

        # 如果正样本太少 ADASYN 可能失败，设保护
        min_class = y.value_counts().min()

        for fs_name in FS_METHODS:
            for imb_name in list(IMBALANCE.keys()):
                # 正样本极少时跳过 ADASYN
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

    # ── 保存结果 ──────────────────────────────────────
    df_all = pd.DataFrame(records)
    csv_path  = os.path.join(OUT_DIR, "all_results.csv")
    xlsx_path = os.path.join(OUT_DIR, "all_results.xlsx")
    df_all.to_csv(csv_path, index=False)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_all.to_excel(writer, sheet_name="All", index=False)

        # 每个数据集一个 sheet
        for ds in df_all["Dataset"].unique():
            df_all[df_all["Dataset"] == ds].to_excel(
                writer, sheet_name=ds[:31], index=False)

        # 最优组合摘要
        best = (df_all.sort_values("F1", ascending=False)
                      .groupby("Dataset").first().reset_index())
        best.to_excel(writer, sheet_name="BestPerDataset", index=False)

    print(f"\n{'='*55}")
    print(f"实验完成！结果已保存：")
    print(f"  CSV  → {csv_path}")
    print(f"  Excel→ {xlsx_path}")
    print(f"总记录数：{len(df_all)}")


if __name__ == "__main__":
    main()
