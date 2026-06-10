import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import pandas as pd
import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from src.preprocessing import preprocess_data

optuna.logging.set_verbosity(optuna.logging.WARNING)

df = pd.read_csv("data/Train.csv")
df = preprocess_data(df)
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def objective(trial):
    params = {
        "iterations":    trial.suggest_int("iterations", 200, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "depth":         trial.suggest_int("depth", 3, 8),
        "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "border_count":  trial.suggest_int("border_count", 32, 255),
        "class_weights": [trial.suggest_float("class_weight_0", 1.0, 5.0), 1.0],
        "random_seed":   42,
        "verbose":       0,
    }

    model = CatBoostClassifier(**params)
    # F1 macro directly penalises low recall on the minority class (rejections)
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    return scores.mean()


if __name__ == "__main__":
    print("Running Optuna search — 60 trials, this takes ~2 minutes...")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=60, show_progress_bar=True)

    best = study.best_params
    # Reconstruct class_weights as a list (Optuna flattens it)
    best_params = {
        "iterations":    best["iterations"],
        "learning_rate": round(best["learning_rate"], 5),
        "depth":         best["depth"],
        "l2_leaf_reg":   round(best["l2_leaf_reg"], 3),
        "border_count":  best["border_count"],
        "class_weights": [round(best["class_weight_0"], 3), 1.0],
        "random_seed":   42,
        "verbose":       0,
    }

    os.makedirs("models", exist_ok=True)
    with open("models/best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    print(f"\nBest AUC (CV):  {study.best_value:.4f}")
    print(f"Best params saved to models/best_params.json")
    print(json.dumps(best_params, indent=2))
