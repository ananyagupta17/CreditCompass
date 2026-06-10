# CreditCompass

An end-to-end ML web application that predicts loan eligibility and explains every decision using SHAP.  
Built with CatBoost, Flask, and deployed on Hugging Face Spaces.

## Live Demo

[Link](https://loan-prediction-h1bh.onrender.com/) (hosted on Render)

---

## Dataset

This project uses the [Loan Prediction Dataset](https://www.kaggle.com/datasets/ninzaami/loan-predication) from Kaggle. It includes information such as applicant income, loan amount, credit history, and property area to determine loan eligibility.

---

## Features

- Cleaned and preprocessed loan application data (614 applicants, 12 features)
- 9 engineered features including `EMI`, `Log_Total_Income`, `Credit_Loan_Ratio`
- **CatBoostClassifier** — chosen over XGBoost and Random Forest based on F1 and AUC on held-out data
- Decision threshold tuned to 0.36 to reduce false negatives (approving bad loans)
- **SHAP explainability** — identifies which features drove each prediction
- **5-fold cross-validated** AUC: 0.76 ± 0.04, F1 macro: 0.73 ± 0.03 (hold-out F1 macro: 0.80)
- Web app built with **Flask**, deployed on **Render**, shows prediction with confidence score

---

## Business Context

Manual loan screening is slow and inconsistent. An automated model that reliably separates eligible applicants from risky ones has direct financial impact:

- **False Positives (bad loan approved):** the bank disburses funds to an applicant likely to default — direct capital loss, typically 5–10× more costly than a missed opportunity.
- **False Negatives (good loan rejected):** a creditworthy customer is turned away — lost interest revenue and customer acquisition cost wasted.

This model is tuned with a **0.36 decision threshold** and **class weights [2, 1]** to prioritise catching bad loans (higher precision on rejections) while still approving 95% of genuinely eligible applicants. At scale across thousands of applications, shifting the false positive rate by even a few percentage points meaningfully reduces portfolio default exposure.

---

## Problem Statement

Financial institutions face challenges in identifying eligible loan applicants.  
This project solves that using supervised learning to classify loan approvals based on attributes like income, credit history, and more.

---

## ML Pipeline

1. **EDA & Preprocessing**

   - Handled missing values
   - Label encoding and one-hot encoding for categorical features
   - Feature engineering (e.g., `Log_Total_Income`, `EMI`, `Credit_Loan_Ratio`)

2. **Model Training**

   - Tried multiple models including Logistic Regression, Random Forest, XGBoost
   - Final model: **CatBoostClassifier** (chosen based on F1-score and AUC)

3. **Evaluation Metrics**
   - Accuracy
   - Precision, Recall, F1-Score
   - AUC-ROC Curve

---

## Tech Stack

| Component    | Technology          |
| ------------ | ------------------- |
| Frontend     | HTML, CSS, Tailwind |
| Backend      | Flask               |
| ML Framework | CatBoost            |
| Deployment   | Render              |

## Web App (Flask)

The web app takes user inputs (loan applicant details) and predicts loan approval instantly.

### 🧾 Loan Application Form

<img src="assets/form.png" alt="Loan Application Form" width="600"/>

### ✅ Prediction Result Page

## <img src="assets/result.png" alt="Prediction Result" width="600"/>
