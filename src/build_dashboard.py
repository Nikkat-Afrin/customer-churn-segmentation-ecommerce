"""Build the interactive segmentation + purchase-intent dashboard (docs/index.html).

Self-contained interactive page combining both halves of the project:

  * K-Means shopper segments in PCA space, with per-segment conversion rates
  * decision-threshold slider for the purchase-intent model (precomputed
    confusion matrices - no server needed)
  * ROC curve and per-class score distributions

Run from the repo root:
    python src/build_dashboard.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from train_pipeline import build_pipeline, load_raw

RNG = 42
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "index.html"

CSS = """
 body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0;
        background: #f4f6f9; color: #1c2733; }
 header { padding: 26px 34px 6px; }
 h1 { margin: 0 0 4px; font-size: 25px; }
 .sub { color: #5c6b7a; font-size: 14px; max-width: 920px; }
 .kpis { display: flex; gap: 14px; padding: 16px 34px 0; flex-wrap: wrap; }
 .kpi { background: white; border-radius: 10px; padding: 12px 20px;
        box-shadow: 0 1px 4px rgba(20,40,80,.08); min-width: 140px; }
 .kpi .v { font-size: 22px; font-weight: 700; color: #0b5fff; }
 .kpi .l { font-size: 12px; color: #5c6b7a; }
 .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(470px, 1fr));
         gap: 18px; padding: 18px 34px 36px; }
 .card { background: white; border-radius: 10px; padding: 6px;
         box-shadow: 0 1px 4px rgba(20,40,80,.08); }
 .wide { grid-column: 1 / -1; }
 footer { padding: 0 34px 26px; color: #8595a5; font-size: 13px; }
"""


def main() -> None:
    X_raw, y = load_raw()

    # ---- segmentation (numeric behavioral features) ------------------------
    numeric = X_raw.select_dtypes(include=[np.number])
    Xs = StandardScaler().fit_transform(numeric)
    km = KMeans(n_clusters=4, random_state=RNG, n_init=10).fit(Xs)
    pcs = PCA(n_components=2, random_state=RNG).fit_transform(Xs)
    seg = pd.DataFrame({"pc1": pcs[:, 0], "pc2": pcs[:, 1],
                        "segment": km.labels_, "converted": y.values})
    rates = seg.groupby("segment")["converted"].agg(["mean", "size"])

    sample = seg.sample(4000, random_state=RNG)
    fig_seg = go.Figure()
    palette = ["#0b5fff", "#e8590c", "#2b8a3e", "#862e9c"]
    for s in sorted(sample["segment"].unique()):
        part = sample[sample["segment"] == s]
        fig_seg.add_trace(go.Scatter(
            x=part["pc1"], y=part["pc2"], mode="markers", name=f"Segment {s}",
            marker=dict(size=5, color=palette[s], opacity=0.55),
            hovertemplate=f"Segment {s} - conv. {rates.loc[s,'mean']:.1%}"))
    fig_seg.update_layout(title="Shopper segments (K-Means, PCA projection, 4k-session sample)",
                          xaxis_title="PC1", yaxis_title="PC2", height=460,
                          margin=dict(l=50, r=20, t=55, b=45))

    fig_rates = go.Figure(go.Bar(
        x=[f"Segment {s}" for s in rates.index], y=rates["mean"],
        text=[f"{v:.1%}<br>({n:,} sessions)" for v, n in zip(rates["mean"], rates["size"])],
        textposition="outside", marker_color=palette[:len(rates)]))
    fig_rates.update_layout(title="Conversion rate by segment",
                            yaxis=dict(title="Sessions ending in purchase", tickformat=".0%"),
                            height=460, margin=dict(l=55, r=20, t=55, b=45))

    # ---- purchase-intent model on held-out test -----------------------------
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_raw, y, test_size=0.2, stratify=y, random_state=RNG)
    pipeline = build_pipeline(X_raw)
    pipeline.set_params(model__n_estimators=150)
    pipeline.fit(X_tr, y_tr)
    proba = pipeline.predict_proba(X_te)[:, 1]
    y_arr = y_te.to_numpy()
    auc = roc_auc_score(y_arr, proba)

    thresholds = np.round(np.linspace(0.05, 0.95, 19), 2)
    fig_thr = go.Figure(); steps = []
    stats = []
    for i, t in enumerate(thresholds):
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_arr, pred).ravel()
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        stats.append((t, prec, rec, f1))
        fig_thr.add_trace(go.Heatmap(
            z=[[fn, tp], [tn, fp]],
            x=["Predicted: no purchase", "Predicted: purchase"],
            y=["Actual: purchase", "Actual: no purchase"],
            text=[[fn, tp], [tn, fp]], texttemplate="%{text}",
            colorscale="Blues", showscale=False, visible=(t == 0.50)))
        steps.append(dict(method="update", label=f"{t:.2f}",
                          args=[{"visible": [j == i for j in range(len(thresholds))]},
                                {"title": f"Threshold {t:.2f} - precision {prec:.2f} · recall {rec:.2f} · F1 {f1:.2f}"}]))
    active = int(np.argmin(np.abs(thresholds - 0.50)))
    t0, p0, r0, f0 = stats[active]
    fig_thr.update_layout(
        sliders=[dict(active=active, currentvalue={"prefix": "Decision threshold: "},
                      steps=steps, pad={"t": 34})],
        title=f"Threshold {t0:.2f} - precision {p0:.2f} · recall {r0:.2f} · F1 {f0:.2f}",
        margin=dict(l=30, r=30, t=60, b=20), height=470)

    fpr, tpr, _ = roc_curve(y_arr, proba)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                 name=f"Random Forest (AUC {auc:.3f})",
                                 line=dict(color="#0b5fff", width=2.5)))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                                 line=dict(dash="dash", color="#98a5b3")))
    fig_roc.update_layout(title="ROC curve (held-out 20% of sessions)",
                          xaxis_title="False positive rate", yaxis_title="True positive rate",
                          height=440, margin=dict(l=50, r=20, t=55, b=45))

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=proba[y_arr == 0], nbinsx=40, name="No purchase",
                                    marker_color="#98a5b3", opacity=0.75))
    fig_dist.add_trace(go.Histogram(x=proba[y_arr == 1], nbinsx=40, name="Purchase",
                                    marker_color="#0b5fff", opacity=0.75))
    fig_dist.update_layout(barmode="overlay", title="Predicted purchase probability by outcome",
                           xaxis_title="P(purchase)", yaxis_title="Sessions",
                           height=440, margin=dict(l=50, r=20, t=55, b=45))

    # ---- page ---------------------------------------------------------------
    kpis = [(f"{len(X_raw):,}", "shopping sessions"), ("4", "K-Means segments"),
            (f"{y.mean():.1%}", "overall conversion"), (f"{auc:.3f}", "test ROC-AUC"),
            (f"{rates['mean'].max():.1%}", "best segment conversion")]
    kpi_html = "".join(f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div></div>'
                       for v, l in kpis)
    charts = []
    order = [fig_seg, fig_rates, fig_thr, fig_roc, fig_dist]
    wide = {2}
    for i, fig in enumerate(order):
        cls = "card wide" if i in wide else "card"
        inner = fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False,
                            div_id=f"chart-{i}")
        charts.append(f'<div class="{cls}">{inner}</div>')
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>E-Commerce Segmentation & Purchase-Intent Dashboard</title><style>{CSS}</style></head><body>
<header><h1>E-Commerce Shopper Segments & Purchase Intent</h1>
<div class="sub">12,330 online shopping sessions (UCI Online Shoppers dataset): K-Means
behavioral segments and a Random-Forest purchase-intent model, with an interactive
decision-threshold explorer showing the precision/recall trade-off a marketing team
would actually tune.</div></header>
<div class="kpis">{kpi_html}</div>
<div class="grid">{''.join(charts)}</div>
<footer>Regenerate: <code>python src/build_dashboard.py</code></footer>
</body></html>"""
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"Dashboard -> {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
