"""Build a HARDER test set that looks more like real seller mistakes.

Our Step 1 violations are obvious (bright colours, big text). Real
mistakes are often subtle, so a model can look great on easy data and
still fail in real life. This set checks that.

For each TEST-split product we save the clean copy plus ONE subtle violation:
  off_white   background replaced by near-white grey/cream (RGB 236-250).
              Still breaks the rule (Amazon needs pure 255) -> nwb=1
  faint_mark  very see-through watermark (alpha 25-55)     -> tlgw=1
  tiny_text   small text in a corner, e.g. "© MyShop"      -> tlgw=1

Only test-split products are used, so no model has seen these products.

Usage:
  python src/make_hard_test.py --processed data/processed --out data/hard_test
"""
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm

from make_violations import WATERMARKS, background_mask, get_font


def off_white(img, rng):
    mask = background_mask(img)
    if mask.mean() < 0.15:
        return None
    v = rng.randint(236, 250)
    tint = rng.choice([(0, 0, 0), (0, -2, -6), (-3, -2, 0)])  # neutral grey, cream, cool
    bg = Image.new("RGB", img.size, tuple(v + t for t in tint))
    m = Image.fromarray(np.uint8(mask) * 255).filter(ImageFilter.GaussianBlur(1))
    return Image.composite(bg, img, m)


def faint_mark(img, rng):
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    font = get_font(rng.randint(16, 28))
    text = rng.choice(WATERMARKS)
    alpha = rng.randint(25, 55)
    for yy in range(10, img.height, 70):
        for xx in range(-20, img.width, 130):
            d.text((xx + (yy // 70 % 2) * 50, yy), text, font=font, fill=(90, 90, 90, alpha))
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


def tiny_text(img, rng):
    img = img.copy()
    d = ImageDraw.Draw(img)
    font = get_font(rng.randint(9, 12))
    text = rng.choice(["© MyShop", "www.myshop.in", "Model #A12", "Made in India", "SKU 4471"])
    l, t, r, b = d.textbbox((0, 0), text, font=font)
    x = rng.choice([4, img.width - (r - l) - 6])
    y = rng.choice([4, img.height - (b - t) - 8])
    g = rng.randint(60, 160)
    d.text((x - l, y - t), text, font=font, fill=(g, g, g))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="data/hard_test")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    src, out = Path(args.processed), Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(src / "labels.csv")
    clean = labels[(labels.split == "test") & (labels.detail == "clean")]

    rows = []
    for r in tqdm(clean.itertuples(), total=len(clean)):
        img = Image.open(src / "images" / r.file).convert("RGB")
        base = dict(source_id=r.source_id, product_type=r.product_type, split="hard")
        img.save(out / "images" / f"{r.source_id}_clean.jpg", quality=92)
        rows.append({**base, "file": f"{r.source_id}_clean.jpg", "tlgw": 0, "nwb": 0, "detail": "clean"})

        kind = rng.choice(["off_white", "faint_mark", "tiny_text"])
        bad = off_white(img, rng) if kind == "off_white" else None
        if bad is None:  # off-white impossible (product fills photo): use a text violation instead
            kind = rng.choice(["faint_mark", "tiny_text"]) if kind == "off_white" else kind
            bad = faint_mark(img, rng) if kind == "faint_mark" else tiny_text(img, rng)
        name = f"{r.source_id}_{kind}.jpg"
        bad.save(out / "images" / name, quality=92)
        rows.append({**base, "file": name, "tlgw": int(kind != "off_white"), "nwb": int(kind == "off_white"), "detail": kind})

    df = pd.DataFrame(rows)
    df.to_csv(out / "labels.csv", index=False)
    print(df.detail.value_counts())
    print(f"saved {len(df)} images -> {out}")


if __name__ == "__main__":
    main()
