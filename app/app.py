import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import preprocess_data
from flask import Flask, render_template, request
import pandas as pd
import pickle
from src.preprocessing import preprocess_data

app = Flask(__name__)

# Load model
with open("models/catboost_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # input from form
    user_input = {
        "Gender": request.form["Gender"],
        "Married": request.form["Married"],
        "Dependents": request.form["Dependents"],
        "Education": request.form["Education"],
        "Self_Employed": request.form["Self_Employed"],
        "ApplicantIncome": float(request.form["ApplicantIncome"]),
        "CoapplicantIncome": float(request.form["CoapplicantIncome"]),
        "LoanAmount": float(request.form["LoanAmount"]),
        "Loan_Amount_Term": float(request.form["Loan_Amount_Term"]),
        "Credit_History": float(request.form["Credit_History"]),
        "Property_Area": request.form["Property_Area"],
        "Loan_ID": "TEMP123"  # dummy placeholder
    }

    df = pd.DataFrame([user_input])
    df = preprocess_data(df)

    y_proba = model.predict_proba(df)[:, 1]
    threshold = 0.36
    prob = float(y_proba[0])
    y_pred = int(prob >= threshold)

    approved = y_pred == 1
    result = "Loan Approved" if approved else "Loan Rejected"
    # Confidence = probability of whichever outcome was predicted
    confidence = round(prob * 100, 1) if approved else round((1 - prob) * 100, 1)

    return render_template("result.html", prediction=result, confidence=confidence, approved=approved)

if __name__ == "__main__":
    app.run(debug=True)
