# Triage IQ — Predicting Hypertension from Emergency Department Triage Data

**Author:** Sairul Behanam (UBD Registration No: 22b6041)  
**Supervisor:** Dr Abdullahi Abubakar Imam  
**Degree:** BSc. in Digital Science, Universiti Brunei Darussalam  
**Date:** 01 May 2026

---

## Project Overview

This project develops and evaluates machine learning models that predict hypertension risk from Emergency Department (ED) triage data at RIPAS Hospital, Brunei Darussalam. Hypertension was the most frequently recorded comorbidity in the triage dataset, yet no systematic screening for it existed in the triage workflow. The goal was to build a model that flags at-risk patients using data already collected during routine care, without requiring any additional clinical steps.

The work is structured in two phases:

- **Phase 1** — Model benchmarking on four public Kaggle datasets to select the best-performing algorithm before committing to the local data
- **Phase 2** — Optimisation and validation on the real RIPAS ED dataset, with SHAP-based interpretability

The final XGBoost model achieved **Recall = 0.89** and **ROC-AUC = 0.945** on the RIPAS dataset.

---

## Repository Structure

```
project/
│
├── README.md                          ← You are here
├── 22b6041_Final_Report.pdf           ← Full dissertation report
│
├── kaggle_data/                       ← Kaggle benchmark datasets (Phase 1 inputs)
│   ├── dataset1.csv
│   ├── dataset2.csv
│   ├── dataset3.csv
│   └── dataset4.csv
│
├── ripas_dataset.csv                  ← RIPAS ED triage data (Phase 2 input)
│
├── dataset1.ipynb                     ← Phase 1: Benchmarking on Dataset 1
├── dataset2.ipynb                     ← Phase 1: Benchmarking on Dataset 2
├── dataset3.ipynb                     ← Phase 1: Benchmarking on Dataset 3
├── dataset4.ipynb                     ← Phase 1: Benchmarking on Dataset 4
│
├── RIPAS_TRIAGE_xgb.ipynb             ← Phase 2: XGBoost model (primary model)
├── RIPAS_TRIAGE_gb.ipynb              ← Phase 2: Gradient Boosting model
└── RIPAS_TRIAGE_hybrid.ipynb          ← Phase 2: Hybrid ensemble (XGBoost + GB)
```

---

## Suggested Reading Order

If you are new to this project, work through the files in this order:

```
22b6041_Final_Report.pdf       ← Start here for the full context and findings
        ↓
dataset2.ipynb                 ← Best Phase 1 notebook to read first (clinical target)
dataset3.ipynb                 ← Second Phase 1 notebook (clinical target)
dataset1.ipynb                 ← Phase 1 with engineered target (note label leakage)
dataset4.ipynb                 ← Phase 1 with engineered target (note label leakage)
        ↓
RIPAS_TRIAGE_xgb.ipynb         ← Primary Phase 2 model — read this first
RIPAS_TRIAGE_gb.ipynb          ← Competing model for comparison
RIPAS_TRIAGE_hybrid.ipynb      ← Ensemble combining the two above
```

---

## Phase 1 — Kaggle Benchmark Notebooks

These four notebooks share an identical pipeline structure. Each one loads a different dataset and runs the same seven classifiers.

### Dataset Overview

| Notebook | File | Target Variable | Target Type | Sample Size |
|---|---|---|---|---|
| `dataset1.ipynb` | `dataset1.csv` | `Hypertension` | Engineered from BP thresholds (JNC-7) | 7,000 (balanced) |
| `dataset2.ipynb` | `dataset2.csv` | `htn` | Clinically recorded diagnosis | 7,000 (balanced) |
| `dataset3.ipynb` | `dataset3.csv` | `hypertension` | Clinically recorded diagnosis | 7,000 (balanced) |
| `dataset4.ipynb` | `dataset4.csv` | `Hypertension` | Engineered from BP thresholds (JNC-7) | 7,000 (balanced) |

> **Important note on Datasets 1 and 4:** The hypertension label in these datasets was derived directly from blood pressure columns using the rule `Systolic BP ≥ 140 OR Diastolic BP ≥ 90`. Because those same BP columns are also model features, several classifiers achieved near-perfect scores — this is **label leakage**, not genuine predictive performance. Results from Datasets 2 and 3 are more trustworthy benchmarks.

### What Each Phase 1 Notebook Does

Each notebook follows the same 16-section structure:

| Section | Description |
|---|---|
| 1. Imports | All libraries loaded upfront |
| 2. Load Dataset | Load CSV, define column groups (numeric, categorical, binary indicators) |
| 3. Balanced Sampling | Downsample to 7,000 rows with 50/50 class split |
| 4. Statistical Feature Analysis | Spearman correlation, Mann–Whitney U test, Chi-square test |
| 5. Pre-Model Feature Importance | Quick XGBoost fit on numeric features to rank importance |
| 6. Train/Test Split | 70/30 stratified split |
| 7. Preprocessing Pipeline | Median imputation + scaling (numeric), mode imputation + one-hot encoding (categorical) |
| 8. Model & Hyperparameter Definitions | 7 classifiers with `RandomizedSearchCV` search spaces |
| 9. Baseline Evaluation | Default parameters, no tuning |
| 10. Hyperparameter Tuning | `RandomizedSearchCV`, 5-fold CV, scoring = ROC-AUC, 30 iterations |
| 11. Before vs After Comparison | Table comparing baseline vs tuned metrics |
| 12. Performance Visualisation | Summary table plot + bar chart |
| 13. Best Hyperparameters | Table of winning parameters per model |
| 14. Confusion Matrix | XGBoost best model on held-out test set |
| 15. SHAP Explainability | Global summary, beeswarm, and local waterfall plots |

### Models Benchmarked

All seven classifiers are evaluated in every Phase 1 notebook:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Gradient Boosting
- Multi-Layer Perceptron (MLP)
- **XGBoost** ← consistently strongest on clinical-target datasets

---

## Phase 2 — RIPAS Triage Notebooks

These three notebooks represent the core contribution of the project. They all operate on the same local hospital dataset and follow a shared pipeline, differing only in which model is used.

### Dataset

`ripas_dataset.csv` — 1,636 de-identified ED patient records from RIPAS Hospital, Brunei Darussalam.

| Feature Group | Examples |
|---|---|
| Demographics | Age, gender |
| Vital signs | Systolic/diastolic BP, heart rate, SpO₂, respiratory rate, temperature, pain score |
| Clinical history | ED visit count, admissions, length of stay |
| Comorbidities | Diabetes mellitus (DM), chronic kidney disease (CKD) |
| Presentation | Site of pain (categorical), chief complaint indicators |
| Outcomes | ICU admission, NIV/ventilation, inotropes, death |
| **Target** | `HTN/CHD` — clinically recorded hypertension flag (binary) |

**Class imbalance:** approximately 96% non-hypertensive, 4% hypertensive — the most significant constraint in this phase.

### Shared Pipeline (all three notebooks)

| Section | Description |
|---|---|
| 1. Imports | All libraries loaded upfront |
| 2. Load Dataset | Load `ripas_dataset.csv` |
| 3. Data Cleaning | Strip `'Y'` suffix from age, cast vitals to numeric, encode binary flags and gender |
| 4. Target Variable | Drop rows with missing `HTN/CHD`; confirm class distribution |
| 5. Train/Test Split | **80/20** stratified split (more training data needed for SMOTE on small dataset) |
| 6. Class Distribution & SMOTE | Visualise imbalance before and after `SMOTE(sampling_strategy=0.8)` |
| 7. Pipeline Definition | Model-specific — see table below |
| 8. Baseline Evaluation | `RepeatedStratifiedKFold(5×5)` + test set evaluation at thresholds 0.5 and 0.35 |
| 9. Hyperparameter Tuning | `RandomizedSearchCV` (broad) → `GridSearchCV` (fine), scoring = recall |
| 10. Threshold Analysis | Sweep from 0.25 to 0.5; recall/precision/F1 table and chart |
| 11. SHAP / Feature Importance | Model-specific interpretability |

### Model-Specific Details

| Notebook | Model | Tuning | SHAP |
|---|---|---|---|
| `RIPAS_TRIAGE_xgb.ipynb` | `XGBClassifier` | RandomCV → GridSearchCV | Yes (TreeExplainer) |
| `RIPAS_TRIAGE_gb.ipynb` | `GradientBoostingClassifier` | RandomCV → GridSearchCV | Feature importances |
| `RIPAS_TRIAGE_hybrid.ipynb` | `VotingClassifier(XGB + GB, soft, weights=[5,1])` | RandomCV (joint param space) | Yes (TreeExplainer on XGB sub-model) |

### Key Results (RIPAS Dataset)

| Model | Stage | Threshold | Accuracy | ROC-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|---|---|
| XGBoost | Baseline | 0.5 | 0.96 | 0.96 | 0.67 | 0.46 | 0.55 |
| **XGBoost** | **RandomCV** | **0.35** | **0.95** | **0.945** | **0.89** | **0.42** | **0.57** |
| XGBoost | GridSearch | 0.5 | 0.95 | 0.96 | 0.67 | 0.40 | 0.50 |
| Gradient Boosting | Baseline | 0.50 | 0.94 | 0.95 | 0.44 | 0.27 | 0.33 |
| Gradient Boosting | RandomCV | 0.35 | 0.89 | 0.92 | 0.89 | 0.24 | 0.37 |
| Gradient Boosting | GridSearch | 0.50 | 0.90 | 0.92 | 0.78 | 0.25 | 0.38 |
| Hybrid Ensemble | Baseline | 0.50 | 0.96 | 0.96 | 0.67 | 0.46 | 0.55 |
| Hybrid Ensemble | Tuned | 0.50 | 0.94 | 0.94 | 0.78 | 0.33 | 0.47 |

> **Selected final model:** XGBoost (RandomCV, threshold = 0.35) — strongest recall with stable ROC-AUC.

### Top Predictors (SHAP — RIPAS Dataset)

Based on SHAP analysis of the final XGBoost model:

1. **Age** — strongest predictor by a wide margin
2. **Diabetes mellitus (DM)** — key comorbidity
3. **Chronic kidney disease (CKD)** — key comorbidity
4. **Site of pain (chest pain)** — moderate influence
5. **SpO₂, respiratory rate, pulse rate** — vital signs contributing at moderate level

Blood pressure readings were notably less dominant than expected, indicating the model captures a broader clinical risk profile rather than re-applying a simple BP threshold rule.

---

## Setup and Dependencies

### Python Version

Python 3.8 or above (developed on Python 3.13.7).

### Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost shap scipy
```

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data handling |
| `matplotlib`, `seaborn` | Visualisation |
| `scikit-learn` | Models, preprocessing, CV, metrics |
| `imbalanced-learn` | SMOTE, ImbPipeline |
| `xgboost` | XGBoost classifier |
| `shap` | Model interpretability |
| `scipy` | Statistical tests (Spearman, Mann–Whitney, Chi-square) |

### Hardware

- Minimum: 8 GB RAM, multi-core CPU (Intel i5 or equivalent)
- Recommended: 16 GB RAM for efficient cross-validation
- GPU: not required (XGBoost and Gradient Boosting are CPU-efficient)

### Running the Notebooks

```bash
# Clone or download the project, then:
jupyter notebook

# Or with JupyterLab:
jupyter lab
```

Run each notebook top-to-bottom.

---

## Methodology Notes

**Why SMOTE is inside the pipeline:** Applying SMOTE before the train/test split would leak synthetic samples into the test set, artificially inflating performance metrics. All three Phase 2 notebooks apply SMOTE exclusively within training folds using `ImbPipeline`.

**Why recall is the primary metric:** In a clinical screening context, a false negative (missed hypertensive patient) carries greater downstream risk than a false positive (unnecessary follow-up check). All hyperparameter tuning uses `scoring='recall'`.

**Why threshold 0.35:** The default 0.5 threshold returned recall of 0.67 for the baseline XGBoost model — too low for a screening tool. A threshold sweep revealed 0.35 as the best balance between recall gain and precision loss, achieving recall 0.89 while keeping precision at 0.42.

**Why 80/20 split (Phase 2) vs 70/30 (Phase 1):** The RIPAS dataset has only 1,636 records. A larger training fraction gives SMOTE more real samples to interpolate from, producing more representative synthetic minorities.

---

## Limitations

- Single-centre dataset (RIPAS Hospital only) — external validation needed before any deployment
- Retrospective data — real-time deployment may present additional challenges
- Structured triage data only — unstructured clinical notes excluded
- Small dataset (n = 1,636) — results should be interpreted with appropriate caution

---

## References

Full references are listed in `22b6041_Final_Report.pdf` (Chapter 6). Key citations:

- Breiman (2001) — Random Forests
- Choi et al. (2017) — RNNs for early heart failure detection
- Chowdhury et al. (2022) — Systematic review of hypertension prediction with ML
- Lundberg & Lee (2017) — SHAP
- Raita et al. (2019) — ML for ED triage outcome prediction
- Smith et al. (2013) — NEWS2 scoring system
