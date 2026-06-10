import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import os
import json
import pandas as pd
import pickle
from src.preprocessing import preprocess_data
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, make_scorer, f1_score
from catboost import CatBoostClassifier

THRESHOLD = 0.50

DEFAULT_PARAMS = {
    "iterations":    300,
    "learning_rate": 0.03,
    "depth":         5,
    "class_weights": [2, 1],
    "random_seed":   42,
    "verbose":       0,
}


def load_params():
    path = "models/best_params.json"
    if os.path.exists(path):
        with open(path) as f:
            params = json.load(f)
        print(f"Loaded tuned params from {path}")
        return params
    print("No best_params.json found — using defaults. Run scripts/tune_model.py first for best results.")
    return DEFAULT_PARAMS


def main():
    df = pd.read_csv("data/Train.csv")
    df = preprocess_data(df)

    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']

    params = load_params()

    # ── 5-fold cross-validation ───────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scorers = {
        'f1_macro': make_scorer(f1_score, average='macro'),
        'roc_auc':  'roc_auc',
    }

    cv_results = cross_validate(CatBoostClassifier(**params), X, y, cv=cv, scoring=scorers, n_jobs=-1)

    print("── 5-Fold Cross-Validation ──────────────────────────────")
    print(f"  F1 macro : {cv_results['test_f1_macro'].mean():.3f}  ±  {cv_results['test_f1_macro'].std():.3f}")
    print(f"  AUC-ROC  : {cv_results['test_roc_auc'].mean():.3f}  ±  {cv_results['test_roc_auc'].std():.3f}")
    print()

    # ── Final train/test split for hold-out evaluation ────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= THRESHOLD).astype(int)

    print("── Hold-out Test Set ────────────────────────────────────")
    print(classification_report(y_test, y_pred))

    with open("models/catboost_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model saved to 'models/catboost_model.pkl'")


if __name__ == "__main__":
    main()
