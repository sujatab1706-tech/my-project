"""Run saved models (and the baselines) on any folder that has labels.csv.

Used in Step 3 to score the hard test set, and reused later by the API.

Usage:
  python src/predict.py --data data/hard_test --models results --out results/hard
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

LABELS = ["tlgw", "nwb"]


def load_batch(paths, size=224):
    return np.stack([np.asarray(Image.open(p).convert("RGB").resize((size, size)), dtype="float32") for p in paths])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", default="results")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from tensorflow import keras  # imported here so --help is fast
    from baselines import nwb_score, tlgw_score

    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data / "labels.csv")
    paths = [data / "images" / f for f in df.file]

    # baselines
    base = df.copy()
    scores = [(nwb_score(Image.open(p)), tlgw_score(Image.open(p))) for p in tqdm(paths, desc="baselines")]
    base["nwb_score"], base["tlgw_score"] = zip(*scores)
    for l in LABELS:
        base[f"{l}_pred"] = (base[f"{l}_score"] >= 0.5).astype(int)
    base.to_csv(out / "baseline_predictions.csv", index=False)

    # CNNs
    x = load_batch(paths)
    for name in ["small", "mobilenet"]:
        model_path = Path(args.models) / f"cnn_{name}.keras"
        if not model_path.exists():
            print("skip (not trained):", model_path)
            continue
        probs = keras.models.load_model(model_path).predict(x, verbose=0)
        part = df.copy()
        for i, l in enumerate(LABELS):
            part[f"{l}_prob"] = probs[:, i]
            part[f"{l}_pred"] = (probs[:, i] >= 0.5).astype(int)
        part.to_csv(out / f"cnn_{name}_predictions.csv", index=False)
    print("saved predictions ->", out)


if __name__ == "__main__":
    main()
