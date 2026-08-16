# Feature Selection and Class Imbalance Handling for Software Defect Prediction

## Title

An empirical comparison of feature selection and class imbalance handling techniques for machine learning-based software defect prediction

## Description

Python implementation of a controlled, full-factorial comparison of 3 feature-selection settings × 4 imbalance-handling strategies × 4 classifiers (48 pipelines), evaluated on 12 NASA MDP software defect datasets under stratified 10-fold cross-validation. Reproduces all 576 experimental results, statistical tests, tables, and figures reported in the paper.

## Dataset Information

- **Source**: NASA Metrics Data Program, distributed via the PROMISE repository
- **URL**: https://promise.site.uml.edu/
- **Version**: D′′ cleaned version prepared by Shepperd et al. (2013), which removes duplicate instances, inconsistent values, and cases with missing data
- **Datasets** (12): CM1, JM1, KC1, KC3, MC1, MC2, MW1, PC1–PC5
- **Preprocessing**: The D′′ cleaning was applied by Shepperd et al. prior to distribution; we applied no further dataset-specific filtering. The categorical defect label was mapped to a binary (0/1) target. Per-fold standardisation (zero mean, unit variance) was applied inside training folds only.
- **Instance counts after preprocessing**: See Table 1 of the manuscript.
- **Third-party data citation**:
  Shepperd M, Song Q, Sun Z, Mair C. 2013. Data quality: some comments on the NASA software defect datasets. IEEE TSE 39(9):1208–1215.

## Code Information

| File | Purpose |
|------|---------|
| `run_experiment_v2.py` | Full 48-pipeline experiment with stratified 10-fold CV; outputs `results/all_results_v2.csv` |
| `generate_figures_v2.py` | Generate Figures 1–7 (boxplots, heatmaps, radar chart) |
| `run_experiment.py` | Original 36-pipeline experiment (v1, for reference) |
| `generate_figures.py` | Original figure generation (v1, for reference) |
| `requirements.txt` | Pinned dependency versions |

## Usage Instructions

```bash
pip install -r requirements.txt
python run_experiment_v2.py      # runs all 576 evaluations (~5 min)
python generate_figures_v2.py    # generates all 7 figures
```

## Requirements

Python >=3.9, scikit-learn, imbalanced-learn, xgboost, numpy, pandas, scipy, matplotlib, seaborn (exact versions pinned in `requirements.txt`)

## Methodology

All data-dependent steps (feature ranking, SMOTE/ADASYN resampling, standardisation, cost-sensitive class-weight computation) are fitted inside each training fold only to prevent data leakage. A fixed random seed (42) is used throughout. SMOTE and ADASYN use k=3 neighbours (reduced from default 5 to accommodate datasets with very small minority classes). Cost-sensitive class weighting uses scikit-learn's `class_weight='balanced'` for LR/SVM/RF and XGBoost's `scale_pos_weight` computed per fold.

## Citations

[Full citation to be added upon publication]

## License

MIT (code). Datasets distributed under their original PROMISE licence terms.
