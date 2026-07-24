# Cancer Type Classification from Gene Expression Data

A machine learning pipeline that classifies cancer type from RNA-Seq gene expression
data, with a deployed Streamlit web app for interactive prediction.

Built as a B.Tech engineering / EPICS community-service project at VIT Bhopal.

**Live demo:** _add your Render URL here once deployed_
`https://<your-app-name>.onrender.com`

---

## Overview

Given a patient's gene expression profile (RNA-Seq), this project predicts which of
5 cancer types the sample belongs to:

- **BRCA** — Breast invasive carcinoma
- **KIRC** — Kidney renal clear cell carcinoma
- **LUAD** — Lung adenocarcinoma
- **PRAD** — Prostate adenocarcinoma
- **COAD** — Colon adenocarcinoma

The pipeline goes from raw 20,531-gene expression data → leakage-safe preprocessing
→ feature selection → model comparison → interpretability analysis → a lightweight
25-gene deployable model with a Streamlit frontend.

**Final result: a 25-gene model matches the accuracy of the full 200-gene model
(99.38% accuracy, 0.9947 macro-F1) — an 8x feature reduction with zero accuracy loss.**

---

## Dataset

- **Source:** [Gene Expression Cancer RNA-Seq](https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq)
  (UCI ML Repository, donated by Fiorini, 2016), derived from the TCGA PAN-Cancer
  RNA-Seq HiSeq project.
- **Shape:** 801 samples × 20,531 gene features
- **Class distribution:**

  | Class | Cancer type | Count |
  |-------|-------------|-------|
  | BRCA  | Breast      | 300   |
  | KIRC  | Kidney      | 146   |
  | LUAD  | Lung        | 141   |
  | PRAD  | Prostate    | 136   |
  | COAD  | Colon       | 78 ← minority class |

- **Data quality:** 0 missing values; genes are anonymized as `gene_0`–`gene_20530`
  in the original release (no public mapping to real gene symbols exists for this
  dataset).
- Raw CSVs are **not committed to git** (75MB, gitignored) — see `raw_data/README.md`
  for how to download and reproduce.

---

## Pipeline

| Stage | Script | What it does |
|-------|--------|---------------|
| 0 | — | Repo init, `.gitignore`, `requirements.txt` |
| 1 | — | Environment setup (venv, dependencies) |
| 2 | — | Data download & source documentation |
| 3 | `src/load_and_eda.py` | Load, merge, EDA — class distribution, missing values, zero-variance genes |
| 4 | `src/preprocessing.py` | Leakage-safe split → drop zero-variance genes → log2(x+1) → StandardScaler → label encode |
| 5 | `src/feature_selection.py` | Compare PCA / SelectKBest / variance-correlation filtering; select k=200 genes |
| 6 | `src/train_models.py` | Tune SVM, Random Forest, XGBoost, MLPClassifier via RandomizedSearchCV (5-fold stratified CV) |
| 7 | `src/evaluate_models.py` | Held-out test set evaluation, confusion matrices, classification reports |
| 8 | `src/interpretability.py` | Feature importances (RF, XGBoost) + SHAP + MLP permutation importance, consolidated gene ranking |
| 9 | `src/train_lightweight_model.py`, `src/build_example_patients.py`, `app.py` | Retrain on top 25 genes, build Streamlit app, deploy |

Each stage is committed to git separately.

### Preprocessing details (Stage 4)

To avoid data leakage, the pipeline:
1. Splits data 80/20 (stratified) **before** any feature statistics are computed
2. Drops zero-variance genes identified from the **train set only** (277 of 20,531)
3. Applies `log2(x + 1)` transform to reduce skew
4. Fits `StandardScaler` on train data only
5. Encodes labels alphabetically: `['BRCA', 'COAD', 'KIRC', 'LUAD', 'PRAD']` → 0–4

Result: 640 train / 161 test samples, 20,254 surviving genes.

### Feature selection (Stage 5)

PCA (95% variance, 447 components), SelectKBest (ANOVA F-test), and variance/correlation
filtering all achieved near-identical cross-validated macro-F1 (~1.0) — this dataset is
genuinely highly separable by cancer type, not a leakage artifact (selectors were fit
inside the CV pipeline). **SelectKBest with k=200** was chosen over PCA for gene-name
interpretability in later stages.

### Model comparison (Stage 6–7)

Tuned via `RandomizedSearchCV` (25 iterations, 5-fold stratified CV, `f1_macro` scoring):

| Model | CV macro-F1 | Test macro-F1 | Test accuracy |
|-------|-------------|----------------|-----------------|
| **MLPClassifier (best)** | 1.0000 | **0.9947** | **99.38%** |
| SVM (RBF) | 0.9987 | — | — |
| Random Forest | 0.9987 | — | — |
| XGBoost | 0.9960 | — | — |

MLP was selected as the best model. On the held-out test set, COAD (minority class)
achieved 100% recall; LUAD was the only class with any misclassifications, likely due
to its smaller sample size.

### Interpretability (Stage 8)

Gene importance was ranked across 5 methods and consolidated by cross-method agreement:
- Random Forest `feature_importances_`
- XGBoost `feature_importances_`
- SHAP `TreeExplainer` (Random Forest)
- SHAP `TreeExplainer` (XGBoost)
- Permutation importance (MLP, since it has no native importance and SHAP
  `KernelExplainer` is too slow at this feature count)

Top 25 genes by agreement count were carried forward to Stage 9.

### Lightweight model & deployment (Stage 9)

An MLP retrained on just the **top 25 consolidated genes** matched the full 200-gene
model exactly on the test set (99.38% accuracy, 0.9947 macro-F1) — validating that
the interpretability-selected genes capture the discriminative signal.

The deployment bundle (`outputs/models/deployment_bundle.joblib`) packages:
- The retrained 25-gene MLP model
- The relevant 25-gene subset of the training-set scaler (mean/scale)
- Label class order

---

## Web App

Built with Streamlit (`app.py`):
- Sliders for all 25 gene expression inputs, ranged to the real dataset's min/max
- "Load a real patient" quick-test feature — pulls actual samples (with known
  ground-truth class) from the dataset per cancer type
- Predict button showing predicted class + confidence bar chart across all 5 classes

Run locally:

```bash
source venv/Scripts/activate   # or venv\Scripts\Activate.ps1 on native PowerShell
pip install -r requirements-render.txt
streamlit run app.py
```

Deployed on [Render](https://render.com) as a free-tier web service.

---

## Project Structure

```
CancerClassification/
├── .gitignore
├── requirements.txt              # full dev environment (training + analysis)
├── requirements-render.txt       # minimal deployment dependencies
├── app.py                        # Streamlit frontend
├── raw_data/
│   ├── README.md                 # how to download the dataset
│   ├── data.csv                  # gitignored (75MB)
│   └── labels.csv                # gitignored
├── src/
│   ├── load_and_eda.py
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   ├── interpretability.py
│   ├── train_lightweight_model.py
│   └── build_example_patients.py
├── notebooks/
└── outputs/
    ├── figures/                  # class distribution, SHAP summary, confusion matrices, etc.
    ├── reports/                  # EDA summary, evaluation summary, consolidated gene rankings
    ├── processed/                # train/test arrays, selected/surviving gene lists
    └── models/                   # scaler, label encoder, feature selector,
                                   # per-model .joblib files, best_model.joblib,
                                   # deployment_bundle.joblib, app_ui_data.json
```

---

## Setup

```bash
git clone <your-repo-url>
cd CancerClassification
python -m venv venv
source venv/Scripts/activate     # git bash on Windows
pip install --upgrade pip
pip install -r requirements.txt
```

Download the dataset per instructions in `raw_data/README.md`, then run stages in order:

```bash
python src/load_and_eda.py
python src/preprocessing.py
python src/feature_selection.py
python src/train_models.py
python src/evaluate_models.py
python src/interpretability.py
python src/train_lightweight_model.py
python src/build_example_patients.py
streamlit run app.py
```

---

## Tech Stack

- **Data & ML:** pandas, numpy, scikit-learn, XGBoost, imbalanced-learn, SHAP
- **Frontend:** Streamlit
- **Deployment:** Render
- **Model serialization:** joblib

---

## Disclaimer

This is a student engineering project trained on a public benchmark dataset for
educational purposes. It is **not a diagnostic tool** and should not be used for
real clinical decision-making.

---

## Acknowledgments

- Dataset: Fiorini, S. (2016). *Gene Expression Cancer RNA-Seq* [Dataset]. UCI
  Machine Learning Repository. https://doi.org/10.24432/C5R88H
- Underlying data from The Cancer Genome Atlas (TCGA) PAN-Cancer RNA-Seq project