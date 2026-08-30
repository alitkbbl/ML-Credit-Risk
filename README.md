# 🏦 ML Credit Risk Pipeline

**Loan Approval Prediction** — an end-to-end machine learning pipeline that
benchmarks classical and boosted-ensemble models on real credit-risk
decisioning, from raw applicant data through a production-style model
comparison and robustness audit.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-blue)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-9cf)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📑 Table of Contents

- [Problem Statement & Dataset Overview](#-problem-statement--dataset-overview)
- [Architecture / Pipeline Overview](#-architecture--pipeline-overview)
- [Key Results & Highlights](#-key-results--highlights)
- [How to Run](#-how-to-run)
- [Project Structure](#-project-structure)
- [Key Takeaways](#-key-takeaways)
- [License](#-license)

---

## 🎯 Problem Statement & Dataset Overview

Financial institutions need to decide, quickly and consistently, whether a
loan application should be **approved** or **rejected**. This project builds
and rigorously compares six classification models on that exact task, using
applicant financial, credit, asset, and demographic data.

**Dataset:** `loan_approval.csv` — tabular loan-application records with:

| Category | Features |
|---|---|
| Credit | `cibil_score` |
| Income & loan terms | `income_annum`, `loan_amount`, `loan_term` |
| Assets | `residential_assets_value`, `commercial_assets_value`, `luxury_assets_value`, `bank_asset_value` |
| Demographic | `no_of_dependents`, `education`, `self_employed` |
| **Target** | `loan_status` (Approved / Rejected) |

The target is moderately imbalanced toward approvals, which shapes several
downstream decisions — resampling strategy, metric choice (F1/AUC over raw
Accuracy), and evaluation depth.

## 🏗 Architecture / Pipeline Overview

```
raw CSV
   │
   ▼
┌─────────────────────────┐
│ 1. EDA                  │  distributions · skewness · correlation heatmap
│                          │  IQR-based outlier detection
└────────────┬─────────────┘
             ▼
┌─────────────────────────┐
│ 2. Preprocessing         │  one-hot encoding · outlier removal
│                          │  feature selection (Pearson ∪ Mutual Information)
└────────────┬─────────────┘
             ▼
┌─────────────────────────┐
│ 3. Train / Test Split    │  80 / 20, stratification-safe, leak-free
└────────────┬─────────────┘
             ▼
┌─────────────────────────┐
│ 4. Imbalance + Scaling   │  SMOTE (train only) → StandardScaler (train-fit)
└────────────┬─────────────┘
             ▼
┌─────────────────────────┐
│ 5. Modeling               │  LogReg · Decision Tree · Random Forest
│    (GridSearchCV, 5-fold) │  KNN · SVM · XGBoost
└────────────┬─────────────┘
             ▼
┌─────────────────────────┐
│ 6. Evaluation             │  Accuracy / Precision / Recall / F1 / ROC-AUC
│                          │  confusion matrices · t-SNE · feature importance
│                          │  robustness check (drop dominant feature)
└─────────────────────────┘
```

## 📊 Key Results & Highlights

Six models were tuned via 5-fold `GridSearchCV` (F1-scored) and evaluated on
a held-out test set:

| Model | Test Accuracy | Test F1 | Test Precision | Test Recall |
|---|---|---|---|---|
| 🏆 **XGBoost** | **0.9777** | **0.9816** | **0.9902** | 0.9731 |
| Random Forest | 0.9683 | 0.9755 | 0.9881 | — |
| Decision Tree | 0.9695 | 0.9748 | — | — |
| SVM (RBF) | 0.9414 | 0.9512 | 0.9663 | 0.9365 |
| KNN | 0.9215 | 0.9327 | 0.9768 | — |
| Logistic Regression | 0.9203 | 0.9324 | 0.9650 | — |

![Model Performance Dashboard](assets/model_performance_dashboard.png)

**Interpretation.**
- **XGBoost is the clear winner** across every headline metric, with CV and
  test scores nearly identical — strong evidence of genuine generalization
  rather than overfitting to the training folds.
- **`cibil_score` (credit score) dominates feature importance** across all
  tree-based models, confirming what the correlation and mutual-information
  analysis in the notebook already showed. A dedicated robustness check
  (dropping this feature and retraining) quantifies exactly how load-bearing
  it is — see the notebook, Section 5.7.
- **No model shows meaningful overfitting** — CV F1 and Test F1 track
  closely for every candidate, which is atypical for a from-scratch student
  pipeline and speaks to a disciplined leak-free split → resample → scale →
  tune workflow.
- Linear and distance-based models (Logistic Regression, KNN) sit closest to
  the random-classifier baseline on the Precision–Recall plot — the true
  decision boundary in this dataset is non-linear, which the tree/ensemble
  models exploit and the linear models cannot.

## 🚀 How to Run

### Prerequisites
- Python 3.10+
- The dataset file `loan_approval.csv` in the project root (not included in
  this repo — source it from your own data provider, e.g. the
  [Kaggle Loan Approval Prediction dataset](https://www.kaggle.com/))

### Installation

```bash
git clone https://github.com/<your-username>/ml-credit-risk-pipeline.git
cd ml-credit-risk-pipeline

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### `requirements.txt`

```
numpy
pandas
seaborn
matplotlib
scikit-learn
imbalanced-learn
xgboost
jupyter
```

### Run the pipeline

```bash
jupyter notebook ml-credit-risk-pipeline.ipynb
```

Run all cells top-to-bottom. The final cell regenerates
`assets/model_performance_dashboard.png` from your own run's results.

## 📁 Project Structure

```
ml-credit-risk-pipeline/
├── ml-credit-risk-pipeline.ipynb    # Full end-to-end pipeline
├── loan_approval.csv                # Dataset (not included — see Prerequisites)
├── assets/
│   └── model_performance_dashboard.png
├── requirements.txt
└── README.md
```

## 🔑 Key Takeaways

- **Best model:** XGBoost — 97.8% test accuracy, 0.982 F1, 0.990 precision.
- **Primary risk driver:** `cibil_score`, overwhelmingly — a strength for
  accuracy, but a fragility for real-world robustness (see the notebook's
  drop-feature audit).
- **Production readiness:** this pipeline is a strong proof of concept, not
  an autonomous decisioning system. Interpretability, fairness auditing, and
  stress-testing against noisy/incomplete real-world data are still needed
  before deployment — the recommended role is **decision support for a human
  underwriter**, not a replacement for one.

## 📄 License

This project is licensed under the MIT License — see the `LICENSE` file for
details.
