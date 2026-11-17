"""Pretend to be sellers uploading photos: send every image in a folder to the API.

Usage (with the API running on port 8000):
  python monitoring/simulate_traffic.py --folder data/hard_test/images --limit 200
"""
import argparse
import random
from pathlib import Path

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8000/predict")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = sorted(Path(args.folder).glob("*.jpg"))
    random.Random(args.seed).shuffle(files)
    for i, f in enumerate(files[: args.limit], 1):
        with f.open("rb") as fh:
            r = requests.post(args.url, files={"file": (f.name, fh, "image/jpeg")}, timeout=30)
        r.raise_for_status()
        if i % 25 == 0:
            print(f"sent {i}")
    print("done:", requests.get(args.url.replace("/predict", "/stats"), timeout=10).json())


if __name__ == "__main__":
    main()
