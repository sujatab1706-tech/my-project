"""Baselines: simple methods with NO training.

Always build a baseline before deep learning. It tells you whether the
complicated model is actually worth it.

  NWB  -> white-border rule (src/rules.py): are the edge pixels white?
  TLGW -> OCR (Optical Character Recognition): can Tesseract read any words?

Each method gives a SCORE between 0 and 1 (how sure it is) and a yes/no
PREDICTION (score >= 0.5). Step 3 uses the scores to pick better thresholds.

Usage:
  python src/baselines.py --data data/processed --out results
Needs the Tesseract program: `apt-get install tesseract-ocr` (Colab/Linux)
or `brew install tesseract` (Mac).
"""
import argparse
import re
from pathlib import Path

import pandas as pd
import pytesseract
from PIL import Image
from tqdm import tqdm

from rules import white_border_fraction


def nwb_score(img: Image.Image) -> float:
    """0 = edges completely white, 1 = no white edge pixels at all.

    The rule in rules.py says 'white' when >= 80% of edge pixels are white,
    so we rescale: 80% white -> score 0.5.
    """
    frac_not_white = 1 - white_border_fraction(img)
    return min(1.0, frac_not_white / 0.4)  # 20% non-white edges -> 0.5


def tlgw_score(img: Image.Image) -> float:
    """Highest OCR confidence (0-1) of any real-looking word in the image."""
    big = img.convert("L").resize((img.width * 2, img.height * 2))  # OCR likes bigger text
    data = pytesseract.image_to_data(big, output_type=pytesseract.Output.DICT, config="--psm 11")
    best = 0.0
    for word, conf in zip(data["text"], data["conf"]):
        word = word.strip()
        # ignore noise: need 2+ letters/digits in a row, e.g. "SALE", "50%"
        if re.search(r"[A-Za-z0-9]{2,}", word) and float(conf) > 0:
            best = max(best, float(conf) / 100)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="results")
    ap.add_argument("--splits", default="val,test")
    args = ap.parse_args()

    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(data / "labels.csv")
    df = labels[labels.split.isin(args.splits.split(","))].copy()

    rows = []
    for r in tqdm(df.itertuples(), total=len(df)):
        img = Image.open(data / "images" / r.file)
        rows.append({"nwb_score": nwb_score(img), "tlgw_score": tlgw_score(img)})
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    df["nwb_pred"] = (df.nwb_score >= 0.5).astype(int)
    df["tlgw_pred"] = (df.tlgw_score >= 0.5).astype(int)
    df.to_csv(out / "baseline_predictions.csv", index=False)
    test = df[df.split == "test"]
    for label in ["tlgw", "nwb"]:
        acc = (test[label] == test[f"{label}_pred"]).mean()
        print(f"{label}: test accuracy {acc:.3f}")
    print("saved", out / "baseline_predictions.csv")


if __name__ == "__main__":
    main()
