"""Create labelled rule-breaking images from clean ones.

Nobody publishes a dataset of "bad" Amazon images, so we make our own:
we take a clean white-background photo and break a rule on purpose.
Because WE broke the rule, we know the correct label for every image.

Phase 1 labels (this script):
  tlgw = 1  text, logo, graphic or watermark added on top of the photo
  nwb  = 1  background is not pure white
(brand and prurience come in phase 2.)

For every clean photo we save:
  - the clean copy                 -> tlgw=0, nwb=0
  - one broken copy, chosen at random from:
      TLGW only, NWB only, or both  -> labels set to match

Usage:
  python src/make_violations.py --raw data/raw --out data/processed
"""
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from tqdm import tqdm

IMG_SIZE = 224  # standard input size for MobileNetV2 and most pretrained CNNs

PHRASES = ["SALE", "50% OFF", "BEST SELLER", "NEW!", "FREE SHIPPING", "HOT DEAL",
           "LIMITED OFFER", "TOP QUALITY", "BUY 1 GET 1", "100% ORIGINAL", "2 PACK"]
WATERMARKS = ["SAMPLE", "WATERMARK", "© MyShop", "www.myshop.com", "PROOF", "DO NOT COPY"]
FONT_FILES = ["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "FreeSansBold.ttf", "Arial Bold.ttf"]


# ---------------------------------------------------------------- helpers
def get_font(size: int) -> ImageFont.ImageFont:
    for name in FONT_FILES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)  # always works with pillow >= 10.1


def random_color(rng: random.Random, alpha: int = 255):
    return (rng.randint(0, 230), rng.randint(0, 230), rng.randint(0, 230), alpha)


def square_on_white(img: Image.Image) -> Image.Image:
    """Pad to a square with white, then resize to IMG_SIZE x IMG_SIZE."""
    img = img.convert("RGB")
    side = max(img.size)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    return canvas.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)


# ---------------------------------------------------------------- TLGW
def add_text_banner(img, rng):
    """Promotional text, sometimes on a coloured box, e.g. '50% OFF'."""
    img = img.convert("RGBA")
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    font = get_font(rng.randint(14, 32))
    text = rng.choice(PHRASES)
    l, t, r, b = d.textbbox((0, 0), text, font=font)
    w, h = r - l, b - t
    x = rng.randint(0, max(0, img.width - w - 8))
    y = rng.choice([rng.randint(0, 30), rng.randint(max(0, img.height - h - 30), img.height - h - 4)])
    if rng.random() < 0.6:  # coloured box behind the text
        d.rectangle([x - 4, y - 4, x + w + 6, y + h + 8], fill=random_color(rng))
        d.text((x - l, y - t), text, font=font, fill=(255, 255, 255, 255))
    else:
        d.text((x - l, y - t), text, font=font, fill=random_color(rng))
    return Image.alpha_composite(img, layer).convert("RGB"), "text"


def add_watermark(img, rng):
    """Semi-transparent text across the photo."""
    img = img.convert("RGBA")
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    font = get_font(rng.randint(18, 36))
    text = rng.choice(WATERMARKS)
    alpha = rng.randint(70, 140)
    gray = rng.randint(60, 160)
    if rng.random() < 0.5:  # one big diagonal watermark
        d.text((img.width * 0.1, img.height * 0.42), text, font=font, fill=(gray, gray, gray, alpha))
        layer = layer.rotate(rng.randint(-35, 35), center=(img.width / 2, img.height / 2))
    else:                   # tiled watermark
        for yy in range(0, img.height, 60):
            for xx in range(-20, img.width, 120):
                d.text((xx + (yy // 60 % 2) * 40, yy), text, font=font, fill=(gray, gray, gray, alpha))
    return Image.alpha_composite(img, layer).convert("RGB"), "watermark"


def add_graphic(img, rng):
    """A badge (circle or star burst) or ribbon, like 'Best Seller' stickers."""
    img = img.convert("RGBA")
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    r = rng.randint(22, 40)
    cx = rng.choice([r + 4, img.width - r - 4])
    cy = rng.choice([r + 4, img.height - r - 4])
    color = random_color(rng)
    if rng.random() < 0.5:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    else:  # star burst
        pts = []
        for k in range(24):
            rad = r if k % 2 == 0 else r * 0.7
            ang = np.pi * k / 12
            pts.append((cx + rad * np.cos(ang), cy + rad * np.sin(ang)))
        d.polygon(pts, fill=color)
    font = get_font(max(9, r // 2))
    text = rng.choice(["NEW", "TOP", "-30%", "HOT", "No.1"])
    l, t, rr, b = d.textbbox((0, 0), text, font=font)
    d.text((cx - (rr - l) / 2 - l, cy - (b - t) / 2 - t), text, font=font, fill=(255, 255, 255, 255))
    return Image.alpha_composite(img, layer).convert("RGB"), "graphic"


def add_tlgw(img, rng):
    fn = rng.choice([add_text_banner, add_text_banner, add_watermark, add_graphic])
    return fn(img, rng)


# ---------------------------------------------------------------- NWB
def background_mask(img: Image.Image, thresh: int = 18) -> np.ndarray:
    """True where the pixel is background.

    Trick: 'flood fill' (like the paint bucket tool) from points on the
    border. Every near-white pixel connected to the border is background;
    white parts INSIDE the product are not reached, so they stay untouched.
    """
    marker = (255, 0, 255)  # magenta: a colour that never appears in the photo after this
    work = img.convert("RGB").copy()
    w, h = work.size
    seeds = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1),
             (w // 2, 0), (w // 2, h - 1), (0, h // 2), (w - 1, h // 2)]
    for s in seeds:
        if min(work.getpixel(s)) >= 255 - thresh:
            ImageDraw.floodfill(work, s, marker, thresh=thresh)
    a = np.asarray(work)
    return (a[..., 0] == 255) & (a[..., 1] == 0) & (a[..., 2] == 255)


def make_background(size, rng):
    w, h = size
    kind = rng.choice(["solid", "gradient", "texture"])
    if kind == "solid":   # e.g. light grey, beige, blue
        c = np.array(random_color(rng)[:3]) if rng.random() < 0.5 else np.array([rng.randint(180, 240)] * 3)
        bg = np.ones((h, w, 3)) * c
    elif kind == "gradient":
        c1, c2 = np.array(random_color(rng)[:3]), np.array(random_color(rng)[:3])
        t = np.linspace(0, 1, h)[:, None, None]
        bg = (c1 * (1 - t) + c2 * t) * np.ones((h, w, 3))
    else:                 # blurry noise, looks like a table or wall
        base = np.array(random_color(rng)[:3])
        noise = np.random.default_rng(rng.randint(0, 10**6)).normal(0, 35, (h // 8, w // 8, 3))
        noise_img = Image.fromarray(np.uint8(np.clip(noise + 128, 0, 255))).resize((w, h), Image.BICUBIC)
        noise = np.asarray(noise_img.filter(ImageFilter.GaussianBlur(4)))
        bg = np.clip(base + (noise.astype(float) - 128), 0, 255)
    return Image.fromarray(np.uint8(bg)), kind


def change_background(img, rng):
    mask = background_mask(img)
    if mask.mean() < 0.15:  # product fills almost the whole photo: nothing to replace
        return None, None
    bg, kind = make_background(img.size, rng)
    # soften the mask edge a little so it doesn't look cut out with scissors
    m = Image.fromarray(np.uint8(mask) * 255).filter(ImageFilter.GaussianBlur(1))
    return Image.composite(bg, img.convert("RGB"), m), f"bg_{kind}"


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    raw, out = Path(args.raw), Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(raw / "metadata.csv")

    # Split by SOURCE photo, so the clean and broken versions of the same product
    # always land in the same split. Otherwise the model could "cheat" by having
    # seen the same product in training (this mistake is called data leakage).
    ids = meta["image_id"].tolist()
    rng.shuffle(ids)
    n = len(ids)
    split_of = {i: ("train" if k < 0.7 * n else "val" if k < 0.85 * n else "test") for k, i in enumerate(ids)}

    rows = []
    for rec in tqdm(meta.itertuples(), total=len(meta)):
        img = square_on_white(Image.open(raw / "images" / f"{rec.image_id}.jpg"))
        base = dict(source_id=rec.image_id, product_type=rec.product_type, split=split_of[rec.image_id])

        img.save(out / "images" / f"{rec.image_id}_clean.jpg", quality=92)
        rows.append({**base, "file": f"{rec.image_id}_clean.jpg", "tlgw": 0, "nwb": 0, "detail": "clean"})

        choice = rng.choice(["tlgw", "nwb", "both"])
        bad, details, labels = img, [], {"tlgw": 0, "nwb": 0}
        if choice in ("nwb", "both"):
            new, d = change_background(bad, rng)
            if new is not None:
                bad, labels["nwb"] = new, 1
                details.append(d)
        if choice in ("tlgw", "both") or labels["nwb"] == 0:  # fall back to TLGW if NWB wasn't possible
            bad, d = add_tlgw(bad, rng)
            labels["tlgw"] = 1
            details.append(d)

        name = f"{rec.image_id}_bad.jpg"
        bad.save(out / "images" / name, quality=92)
        rows.append({**base, "file": name, **labels, "detail": "+".join(details)})

    df = pd.DataFrame(rows)
    df.to_csv(out / "labels.csv", index=False)
    print(df.groupby("split")[["tlgw", "nwb"]].agg(["sum", "count"]))
    print(f"saved {len(df)} images + labels.csv -> {out}")


if __name__ == "__main__":
    main()
