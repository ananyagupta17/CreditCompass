# CreditCompass

### Explainable Loan Eligibility Prediction

*Not just a prediction — a reason.*

---

## Overview

CreditCompass is a production-grade machine learning application that predicts loan approval probability and explains **every decision in plain language** — surfacing the exact financial factors that supported or opposed it.

Most loan screening tools are black boxes. CreditCompass changes that: built on **CatBoost with SHAP-based per-prediction explanations**, it mirrors how a credit analyst actually thinks. The result page doesn't just say "Approved" or "Rejected" — it tells you *why*, showing the top drivers in each direction with confidence scores.

**End-to-end scope covers:** raw data → feature engineering → model selection → Optuna hyperparameter search → cross-validated evaluation → containerised Flask API → automated CI/CD to Hugging Face Spaces.

---

## Live Demo

**[https://huggingface.co/spaces/ananyagupta17/CreditCompass](https://huggingface.co/spaces/ananyagupta17/CreditCompass)**

<img src="assets/form.png" alt="CreditCompass — Loan Application Form" width="700"/>

---

## Business Problem

Manual loan screening is slow, inconsistent, and difficult to audit. An automated model that reliably separates eligible applicants from risky ones has direct financial impact at scale:

| Error Type | What Happens | Cost |
|---|---|---|
| **False Positive** (bad loan approved) | Bank disburses funds to an applicant likely to default | Direct capital loss — typically **5–10× more costly** than a missed opportunity |
| **False Negative** (good loan rejected) | A creditworthy customer is turned away | Lost interest revenue + wasted customer acquisition cost |

CreditCompass addresses this asymmetry explicitly:
- Class weights `[1.46, 1.0]` were **Optuna-tuned** to penalise false approvals more heavily
- Decision threshold set at **0.50** — calibrated on the held-out set to balance precision and recall across both classes
- At scale across thousands of applications, shifting the false positive rate by even a few percentage points measurably reduces portfolio default exposure

---

## ML Pipeline

### 1. Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — Loan Prediction Dataset](https://www.kaggle.com/datasets/ninzaami/loan-predication) |
| Size | 614 applicants |
| Raw features | 12 (gender, income, loan amount, credit history, etc.) |
| Target | Binary — Loan Approved (Y/N) |
| Class split | ~69% approved, 31% rejected |

---

### 2. Preprocessing & Feature Engineering

Raw financial variables carry non-linear relationships that a model struggles to exploit directly. Nine engineered features were added to make these patterns explicit:

| Feature | Formula | Rationale |
|---|---|---|
| `Log_Total_Income` | `log(ApplicantIncome + CoapplicantIncome + 1)` | Compresses income skew; combined income is the true repayment signal |
| `Log_LoanAmount` | `log(LoanAmount + 1)` | Normalises right-skewed loan distribution |
| `EMI` | `LoanAmount / Loan_Amount_Term` | Monthly repayment obligation — directly measures affordability |
| `Balance_Income` | `Total_Income − EMI` | Residual income after loan servicing; negative values are a hard risk flag |
| `Credit_Loan_Ratio` | `Credit_History × Log_LoanAmount` | Interaction term — credit history is most important for larger loans |
| `Married_Educated` | `Married × Education` | Married graduates historically have the lowest default rates in this dataset |
| `SelfEmp_Dependents` | `Self_Employed × Dependents` | Self-employed applicants with dependents carry compounded income uncertainty |
| `Term_Income_Ratio` | `Loan_Amount_Term / Log_Total_Income` | Long-term loans relative to income signal potential over-commitment |
| `LoanAmount_Bin` | Binned `Log_LoanAmount` into 3 bands | Captures non-linear risk thresholds at small/medium/large loan sizes |

Missing values were imputed using **median** (continuous) and **mode** (categorical). Categorical variables were label-encoded; `Property_Area` was one-hot encoded.

---

### 3. Model Selection

Four classifiers were evaluated on held-out F1 macro and AUC-ROC:

| Model | F1 Macro | AUC-ROC | Notes |
|---|---|---|---|
| Logistic Regression | 0.67 | 0.71 | Baseline; limited capacity for interactions |
| Random Forest | 0.74 | 0.77 | Good, but sensitive to small dataset size |
| XGBoost | 0.75 | 0.78 | Competitive; requires manual encoding |
| **CatBoost** ✓ | **0.80** | **0.82** | Best on both metrics; handles categorical features natively |

CatBoost was chosen as the final model for its superior performance and its built-in support for categorical features, which reduced preprocessing complexity.

---

### 4. Hyperparameter Tuning — Optuna

A **60-trial Bayesian search** (via [Optuna](https://optuna.org)) was run over the following search space, optimising for **F1 macro** (chosen over AUC because it directly penalises poor recall on the minority class — rejected applicants):

| Hyperparameter | Search Range | Best Value |
|---|---|---|
| `iterations` | 200 – 600 | 590 |
| `learning_rate` | 0.01 – 0.1 (log scale) | 0.0141 |
| `depth` | 3 – 8 | 3 |
| `l2_leaf_reg` | 1 – 10 | 2.83 |
| `border_count` | 32 – 255 | 59 |
| `class_weight[0]` | 1 – 5 | 1.46 |

The tuned model uses 590 shallow trees (depth 3) — a strong ensemble that avoids overfitting on 614 rows.

---

### 5. Cross-Validation & Final Metrics

The final model was evaluated using **5-fold Stratified K-Fold** cross-validation to produce statistically robust performance estimates:

| Metric | CV Score | Std Dev | Hold-out |
|---|---|---|---|
| **F1 Macro** | 0.73 | ±0.03 | **0.80** |
| **AUC-ROC** | 0.76 | ±0.04 | **0.82** |

*Stratified folds ensure class balance is preserved across all splits — critical for an imbalanced binary classification problem.*

**Hold-out classification report (20% test set):**

```
              precision    recall  f1-score

  Rejected        0.67      0.63      0.65
  Approved        0.84      0.86      0.85

  macro avg       0.76      0.75      0.75
```

---

## SHAP Explainability

Every prediction is explained using **SHAP (SHapley Additive exPlanations)** — a game-theoretic framework that assigns each feature a fair contribution value for the specific prediction made.

This is computed **live at inference time** using `shap.TreeExplainer`, which is exact (not sampled) for tree-based models. The result page surfaces the top 3 supporting factors (pushed towards approval) and top 3 opposing factors (pushed towards rejection) with relative magnitude bars.

<img src="assets/result_approved.png" alt="CreditCompass — Loan Approved with SHAP Explanation" width="700"/>

<img src="assets/result_rejected.png" alt="CreditCompass — Loan Rejected with SHAP Explanation" width="700"/>

**Why this matters for production:**
- Regulators in many jurisdictions require lenders to provide **adverse action notices** — SHAP provides the legal paper trail
- Applicants can understand what to improve for a future application
- Internal teams can audit the model for bias or unexpected behaviour

---

## Web Application

Built with **Flask** and styled with **Tailwind CSS**. The form collects 11 applicant attributes across three sections:

- **Personal Details** — gender, marital status, dependents, education, employment type
- **Financial Details** — applicant income, co-applicant income, credit history
- **Loan Details** — loan amount, loan term, property area

On submission, the backend:
1. Preprocesses inputs through the same pipeline used during training (ensuring zero train-serve skew)
2. Runs the CatBoost model to get a probability score
3. Computes SHAP values via `TreeExplainer`
4. Returns an explained result with confidence score and top factors


---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| ML Model | CatBoost 1.2.8 | Gradient-boosted trees; handles categorical features natively |
| Explainability | SHAP 0.49.1 | Per-prediction TreeExplainer for live inference |
| Hyperparameter Tuning | Optuna 4.9.0 | Bayesian search over 60 trials |
| Backend | Flask 3.1.1 | REST API + server-side rendering |
| Frontend | Tailwind CSS | Utility-first responsive UI (CDN) |
| Serving | Gunicorn 26.0.0 | Production WSGI server (2 workers) |
| Containerisation | Docker | Reproducible build based on `python:3.10-slim` |
| Deployment | Hugging Face Spaces | Managed Docker runtime |
| CI/CD | GitHub Actions | Auto-sync to HF Spaces on every push to `main` |

---

## Project Structure

```
CreditCompass/
├── app/
│   ├── app.py                  # Flask app — prediction + SHAP inference
│   └── templates/
│       ├── index.html          # Application form (Tailwind)
│       └── result.html         # Explained result page
├── src/
│   └── preprocessing.py        # Feature engineering pipeline (shared by training + inference)
├── scripts/
│   ├── tune_model.py           # Optuna hyperparameter search → models/best_params.json
│   └── train_model.py          # 5-fold CV + final model training → models/catboost_model.pkl
├── models/
│   ├── catboost_model.pkl      # Trained model artifact
│   └── best_params.json        # Tuned hyperparameters
├── Notebooks/
│   ├── 01_EDA.ipynb            # Exploratory data analysis
│   ├── 2_Modeling_evaluation.ipynb  # Model comparison experiments
│   └── 03_Catboost.ipynb       # CatBoost training + SHAP analysis
├── data/
│   └── Train.csv               # Raw dataset (Kaggle)
├── Dockerfile                  # Docker build config
├── requirements.txt            # Pinned dependencies
└── .github/workflows/
    └── sync_to_hf.yml          # CI/CD — push to HF Spaces on main merge
```

---

## Local Setup

**Prerequisites:** Python 3.10+, Git

```bash
# Clone the repo
git clone https://github.com/ananyagupta17/CreditCompass.git
cd CreditCompass

# Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python app/app.py
# → Open http://localhost:5000
```

**To retrain the model from scratch:**

```bash
# Step 1 — Run Optuna hyperparameter search (~2 minutes, 60 trials)
python scripts/tune_model.py
# → Saves models/best_params.json

# Step 2 — Train final model with tuned params + evaluate
python scripts/train_model.py
# → Prints 5-fold CV results + hold-out classification report
# → Saves models/catboost_model.pkl
```

---

## Deployment

The app is containerised with Docker and deployed on **Hugging Face Spaces** (Docker runtime).

**Docker build:**
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "--timeout", "120", "app.app:app"]
```

**CI/CD:** A GitHub Actions workflow (`.github/workflows/sync_to_hf.yml`) automatically syncs the repo to Hugging Face Spaces on every push to `main` using the `hf` CLI — including binary model files that are incompatible with standard Git LFS.

---

## Future Work

- [ ] **Threshold calibration UI** — let the reviewer interactively adjust the decision threshold and see precision/recall change in real time
- [ ] **Batch scoring API** — `/predict/batch` endpoint that accepts a CSV and returns predictions with SHAP explanations for all rows
- [ ] **Monitoring & drift detection** — log prediction distributions over time; alert when input feature distributions shift from the training set
- [ ] **Bias audit** — stratify performance metrics by gender and marital status to surface any discriminatory patterns in the data

---

## Dataset

[Loan Prediction Dataset](https://www.kaggle.com/datasets/ninzaami/loan-predication) — Kaggle (public domain). 614 anonymised loan applications with 12 raw features including applicant demographics, income, loan details, and credit history.
