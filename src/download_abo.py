"""Download a sample of clean Amazon product photos from Amazon Berkeley Objects (ABO).

ABO is a free public dataset (licence: CC BY-NC 4.0, fine for learning).
Its files are on a public web address, so no AWS account is needed.

What this script does:
  1. Downloads the listing metadata (product type, brand, main image id).
  2. Keeps only each product's MAIN image (the one that must follow the rules).
  3. Downloads a random sample of those images in "small" size (max 256 px).
  4. Keeps only photos that already have a white background, so we start
     from images we are confident are "clean".

Usage:
  python src/download_abo.py --n 3000 --out data/raw
"""
import argparse
import ast
import gzip
import io
import json
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

from rules import is_white_background

BASE = "https://amazon-berkeley-objects.s3.amazonaws.com"


def first_value(field):
    """ABO stores text like [{'language_tag': 'en_US', 'value': 'Nike'}]. Return 'Nike'."""
    if not field:
        return None
    if isinstance(field, str):
        field = ast.literal_eval(field)
    for item in field:
        if item.get("language_tag", "en_US").startswith("en"):
            return item["value"]
    return field[0]["value"]


def load_listings(n_files: int) -> pd.DataFrame:
    rows = []
    for i in range(n_files):
        url = f"{BASE}/listings/metadata/listings_{i}.json.gz"
        print("downloading", url)
        raw = requests.get(url, timeout=60).content
        for line in gzip.decompress(raw).decode("utf-8").splitlines():
            d = json.loads(line)
            if "main_image_id" not in d:
                continue
            rows.append({
                "item_id": d["item_id"],
                "image_id": d["main_image_id"],
                "brand": first_value(d.get("brand")),
                "product_type": first_value(d.get("product_type")),
            })
    return pd.DataFrame(rows).drop_duplicates("image_id")


def load_image_paths() -> pd.DataFrame:
    url = f"{BASE}/images/metadata/images.csv.gz"
    print("downloading", url)
    return pd.read_csv(io.BytesIO(requests.get(url, timeout=120).content), compression="gzip")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="how many images to try downloading")
    ap.add_argument("--listing-files", type=int, default=4, help="ABO has 16 listing files; 4 is plenty")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    listings = load_listings(args.listing_files)
    paths = load_image_paths()
    df = listings.merge(paths, on="image_id")
    df = df.sample(n=min(args.n, len(df)), random_state=args.seed)
    print(f"{len(df)} main images selected, downloading...")

    kept = []
    for row in tqdm(df.itertuples(), total=len(df)):
        dest = out / "images" / f"{row.image_id}.jpg"
        try:
            if not dest.exists():
                r = requests.get(f"{BASE}/images/small/{row.path}", timeout=30)
                r.raise_for_status()
                dest.write_bytes(r.content)
            img = Image.open(dest)
            if min(img.size) < 100:          # too tiny to be useful
                dest.unlink()
                continue
            if not is_white_background(img):  # we only want clean starting images
                dest.unlink()
                continue
            kept.append(row._asdict())
        except Exception as e:  # network hiccup or broken file: skip it
            print("skip", row.image_id, e)

    meta = pd.DataFrame(kept).drop(columns=["Index"])
    meta.to_csv(out / "metadata.csv", index=False)
    print(f"kept {len(meta)} clean white-background images -> {out}")


if __name__ == "__main__":
    main()
