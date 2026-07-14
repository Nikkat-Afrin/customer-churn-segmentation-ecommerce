"""Train and persist a purchase-intent scoring pipeline.

Unlike the exploratory comparison script, this builds a single sklearn
Pipeline (preprocessing + model) so that scoring can never drift from
training-time preprocessing:

    ColumnTransformer
      - numeric columns  -> StandardScaler
      - categoricals     -> OneHotEncoder(handle_unknown="ignore")
    -> RandomForestClassifier(class_weight="balanced_subsample")

Outputs:
    models/purchase_intent_rf.joblib   (the whole fitted pipeline)
    reports/metrics.json               (CV + held-out test metrics)

Usage:
    python src/train_pipeline.py [--folds 5] [--test-size 0.2]
"""

import argparse
import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RNG = 42
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "purchase_intent_rf.joblib"


def load_raw():
    X = pd.read_csv(ROOT / "data" / "Project2_Data.csv")
    y = pd.read_csv(ROOT / "data" / "Project2_Data_Labels.csv").iloc[:, 0]
    y = y.astype(str).str.upper().map({"TRUE": 1, "FALSE": 0}).astype(int)
    return X, y


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    categorical = list(X.select_dtypes(include=["object", "bool"]).columns)
    numeric = [c for c in X.columns if c not in categorical]
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced_subsample",
        random_state=RNG, n_jobs=-1)
    return Pipeline([("preprocess", preprocess), ("model", model)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    X, y = load_raw()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=RNG)

    pipeline = build_pipeline(X)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=RNG)
    cv_auc = cross_val_score(pipeline, X_tr, y_tr, cv=skf,
                             scoring="roc_auc", n_jobs=-1)
    print(f"CV ROC-AUC: {cv_auc.mean():.4f} +/- {cv_auc.std():.4f}")

    pipeline.fit(X_tr, y_tr)
    proba = pipeline.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "trained_on": str(date.today()),
        "n_samples": int(len(y)),
        "positive_rate": round(float(y.mean()), 4),
        "cv": {"folds": args.folds,
               "roc_auc_mean": round(float(cv_auc.mean()), 4),
               "roc_auc_std": round(float(cv_auc.std()), 4)},
        "test": {
            "roc_auc": round(float(roc_auc_score(y_te, proba)), 4),
            "pr_auc": round(float(average_precision_score(y_te, proba)), 4),
            "accuracy": round(float(accuracy_score(y_te, pred)), 4),
            "precision": round(float(precision_score(y_te, pred)), 4),
            "recall": round(float(recall_score(y_te, pred)), 4),
            "f1": round(float(f1_score(y_te, pred)), 4),
        },
    }
    print(json.dumps(metrics["test"], indent=2))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved {MODEL_PATH} and reports/metrics.json")


if __name__ == "__main__":
    main()
