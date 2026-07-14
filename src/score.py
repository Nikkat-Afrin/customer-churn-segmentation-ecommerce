"""Score shopping sessions with the persisted purchase-intent pipeline.

Usage:
    python src/score.py data/Project2_Data.csv --out scored_sessions.csv
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "purchase_intent_rf.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--out", type=Path, default=Path("scored_sessions.csv"))
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"{args.model} not found - run `python src/train_pipeline.py` first.")

    pipeline = joblib.load(args.model)
    df = pd.read_csv(args.input_csv)
    proba = pipeline.predict_proba(df)[:, 1]

    out = df.copy()
    out["purchase_probability"] = proba.round(4)
    out["purchase_prediction"] = (proba >= 0.5).astype(int)
    out.to_csv(args.out, index=False)
    print(f"Scored {len(out)} sessions -> {args.out} "
          f"({int(out['purchase_prediction'].sum())} predicted buyers)")


if __name__ == "__main__":
    main()
