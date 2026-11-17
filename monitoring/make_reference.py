"""Build the REFERENCE data for monitoring: what "normal" looks like.

We run the deployed predictor over the validation images, which come from the
same kind of data the model was trained on, and save the same columns the
API logs. Monitoring then compares live traffic against this file.

Usage:
  python monitoring/make_reference.py --data data/processed --split val --out monitoring/reference.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from inference import Predictor  # noqa: E402


def rows_for(predictor, paths):
    out = []
    for p in tqdm(paths):
        r = predictor.predict(Image.open(p))
        out.append({
            "filename": Path(p).name,
            "tlgw_prob": r["tlgw"]["probability"], "nwb_prob": r["nwb"]["probability"],
            "tlgw_flag": int(r["tlgw"]["flag"]), "nwb_flag": int(r["nwb"]["flag"]),
            "nwb_cnn_flag": int(r["nwb"]["cnn_flag"]), "nwb_rule_flag": int(r["nwb"]["rule_flag"]),
            **r["image"], "latency_ms": r["latency_ms"],
        })
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="monitoring/reference.csv")
    ap.add_argument("--model", default="models/cnn_mobilenet.keras")
    ap.add_argument("--thresholds", default="models/thresholds.json")
    args = ap.parse_args()

    labels = pd.read_csv(Path(args.data) / "labels.csv")
    files = labels[labels.split == args.split].file
    df = rows_for(Predictor(args.model, args.thresholds), [Path(args.data) / "images" / f for f in files])
    df.to_csv(args.out, index=False)
    print(f"saved {len(df)} reference rows -> {args.out}")


if __name__ == "__main__":
    main()
