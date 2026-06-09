"""
E-commerce customer segmentation + purchase-intent prediction (UCI Online Shoppers).

Two analyses on 12,330 sessions:
  1. SEGMENTATION  — K-Means on scaled behavioral features (silhouette-checked),
                     visualized in 2-D via PCA.
  2. PREDICTION    — classify whether a session converts (Revenue=True, ~15.5%)
                     with Logistic Regression vs Random Forest (imbalance-aware).

Writes reports/model_comparison.md and reports/figures/*.png.
Run from repo root:  python src/segmentation_and_prediction.py
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, ConfusionMatrixDisplay)

warnings.filterwarnings("ignore")
RNG = 42
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def load():
    X = pd.read_csv(ROOT / "data" / "Project2_Data.csv")
    y = pd.read_csv(ROOT / "data" / "Project2_Data_Labels.csv").iloc[:, 0].astype(int)
    X = pd.get_dummies(X, columns=list(X.select_dtypes(include=["object", "bool"]).columns),
                       drop_first=True, dtype=int)
    return X.fillna(0), y


def segmentation(Xs):
    sils = {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=RNG, n_init=10).fit(Xs)
        sils[k] = silhouette_score(Xs, km.labels_, sample_size=3000, random_state=RNG)
    best_k = max(sils, key=sils.get)
    km = KMeans(n_clusters=best_k, random_state=RNG, n_init=10).fit(Xs)
    pcs = PCA(n_components=2, random_state=RNG).fit_transform(Xs)
    plt.figure(figsize=(7, 6))
    plt.scatter(pcs[:, 0], pcs[:, 1], c=km.labels_, cmap="tab10", s=6, alpha=0.5)
    plt.title(f"K-Means segments (k={best_k}, silhouette={sils[best_k]:.3f}) — PCA view")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout(); plt.savefig(FIG / "segments_pca.png", dpi=120); plt.close()
    print(f"Segmentation: silhouettes={ {k: round(v,3) for k,v in sils.items()} } -> best k={best_k}")
    return best_k, sils[best_k]


def prediction(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RNG)
    sc = StandardScaler().fit(X_tr)
    X_tr = pd.DataFrame(sc.transform(X_tr), columns=X.columns, index=X_tr.index)
    X_te = pd.DataFrame(sc.transform(X_te), columns=X.columns, index=X_te.index)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                random_state=RNG, n_jobs=-1),
    }
    rows, roc_store, fitted = [], {}, {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        proba = m.predict_proba(X_te)[:, 1]; pred = m.predict(X_te)
        roc_store[name] = roc_curve(y_te, proba)
        rows.append({"Model": name, "Accuracy": accuracy_score(y_te, pred),
                     "Precision": precision_score(y_te, pred, zero_division=0),
                     "Recall": recall_score(y_te, pred, zero_division=0),
                     "F1": f1_score(y_te, pred, zero_division=0),
                     "ROC-AUC": roc_auc_score(y_te, proba)})
        fitted[name] = m
    res = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    best = res.iloc[0]["Model"]
    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr, _) in roc_store.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_te, fitted[name].predict_proba(X_te)[:,1]):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC — Purchase-intent prediction"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(FIG / "roc_curves.png", dpi=120); plt.close()
    ConfusionMatrixDisplay(confusion_matrix(y_te, fitted[best].predict(X_te)),
                           display_labels=["No buy", "Purchase"]).plot(cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix — {best} (test)")
    plt.tight_layout(); plt.savefig(FIG / "confusion_matrix_best.png", dpi=120); plt.close()
    return res, best


def main():
    X, y = load()
    print(f"sessions={len(y)}  features={X.shape[1]}  purchase_rate={y.mean():.3f}")
    Xs = StandardScaler().fit_transform(X)
    best_k, sil = segmentation(Xs)
    res, best = prediction(X, y)
    (ROOT / "reports").mkdir(exist_ok=True)
    cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    fmt = lambda v: v if isinstance(v, str) else f"{v:.3f}"
    lines = [f"# E-commerce: segmentation (k={best_k}, silhouette={sil:.3f}) + purchase prediction", "",
             "| " + " | ".join(cols) + " |", "|" + "|".join(["---"]*len(cols)) + "|"]
    for _, r in res.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    (ROOT / "reports" / "model_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Best classifier: {best}\nWritten to {ROOT/'reports'}")


if __name__ == "__main__":
    main()
