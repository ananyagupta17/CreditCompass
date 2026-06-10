import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shap
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from src.preprocessing import preprocess_data

app = Flask(__name__)


def format_inr(n):
    s = str(int(n))
    if len(s) <= 3:
        return s
    result = s[-3:]
    s = s[:-3]
    while s:
        result = s[-2:] + ',' + result
        s = s[:-2]
    return result.lstrip(',')

app.jinja_env.filters['inr'] = format_inr

with open("models/catboost_model.pkl", "rb") as f:
    model = pickle.load(f)

explainer = shap.TreeExplainer(model)

FEATURE_LABELS = {
    'Credit_History':       'Credit History',
    'Credit_Loan_Ratio':    'Credit-Loan Score',
    'Log_Total_Income':     'Total Income',
    'Log_LoanAmount':       'Loan Amount',
    'EMI':                  'Monthly EMI',
    'Balance_Income':       'Residual Income after EMI',
    'Loan_Amount_Term':     'Loan Term',
    'Gender':               'Gender',
    'Married':              'Marital Status',
    'Dependents':           'No. of Dependents',
    'Education':            'Education Level',
    'Self_Employed':        'Employment Type',
    'Property_Area_Rural':  'Rural Property',
    'Property_Area_Semiurban': 'Semi-urban Property',
    'Property_Area_Urban':  'Urban Property',
    'Married_Educated':     'Married & Graduate',
    'SelfEmp_Dependents':   'Self-employed with Dependents',
    'Term_Income_Ratio':    'Loan Term vs Income',
    'LoanAmount_Bin':       'Loan Size Band',
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    loan_amount_rupees = float(request.form["LoanAmount"])
    user_input = {
        "Gender":           request.form["Gender"],
        "Married":          request.form["Married"],
        "Dependents":       request.form["Dependents"],
        "Education":        request.form["Education"],
        "Self_Employed":    request.form["Self_Employed"],
        "ApplicantIncome":  float(request.form["ApplicantIncome"]),
        "CoapplicantIncome": float(request.form["CoapplicantIncome"]),
        "LoanAmount":       loan_amount_rupees / 1000,
        "Loan_Amount_Term": float(request.form["Loan_Amount_Term"]),
        "Credit_History":   float(request.form["Credit_History"]),
        "Property_Area":    request.form["Property_Area"],
        "Loan_ID":          "TEMP123",
    }

    df = pd.DataFrame([user_input])
    df_processed = preprocess_data(df)

    prob = float(model.predict_proba(df_processed)[:, 1][0])
    approved = prob >= 0.50
    confidence = round(prob * 100, 1) if approved else round((1 - prob) * 100, 1)

    # Per-prediction SHAP factors
    shap_vals = explainer.shap_values(df_processed)
    sv = shap_vals[0] if isinstance(shap_vals, list) else shap_vals[0]

    feature_names = df_processed.columns.tolist()
    raw = [(FEATURE_LABELS.get(f, f), float(v)) for f, v in zip(feature_names, sv)]

    max_magnitude = max(abs(v) for _, v in raw) or 1

    supporting = [
        {"name": name, "width": round(abs(val) / max_magnitude * 100)}
        for name, val in sorted(raw, key=lambda x: x[1], reverse=True)
        if val > 0.001
    ][:3]

    opposing = [
        {"name": name, "width": round(abs(val) / max_magnitude * 100)}
        for name, val in sorted(raw, key=lambda x: x[1])
        if val < -0.001
    ][:3]

    return render_template(
        "result.html",
        approved=approved,
        confidence=confidence,
        supporting=supporting,
        opposing=opposing,
        applicant_income=int(user_input["ApplicantIncome"]),
        loan_amount=int(loan_amount_rupees),
        loan_term=int(user_input["Loan_Amount_Term"]),
        credit_history="Yes" if user_input["Credit_History"] == 1.0 else "No",
    )


if __name__ == "__main__":
    app.run(debug=True)
