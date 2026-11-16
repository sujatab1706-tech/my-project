"""Simple, hand-written rules (no machine learning).

These are our *baselines*: later we check whether the CNN actually beats them.
"""
import numpy as np
from PIL import Image


def white_border_fraction(img: Image.Image, border: int = 4, tol: int = 12) -> float:
    """Share of pixels on the image edges that are (almost) pure white.

    Amazon main images must have a pure white background (RGB 255,255,255).
    JPEG compression nudges 255 down to 250-254, so we allow a small
    tolerance `tol`.
    """
    a = np.asarray(img.convert("RGB"), dtype=np.int16)
    edges = np.concatenate([
        a[:border].reshape(-1, 3),        # top rows
        a[-border:].reshape(-1, 3),       # bottom rows
        a[:, :border].reshape(-1, 3),     # left columns
        a[:, -border:].reshape(-1, 3),    # right columns
    ])
    near_white = (edges >= 255 - tol).all(axis=1)
    return float(near_white.mean())


def is_white_background(img: Image.Image, threshold: float = 0.80) -> bool:
    """Rule: the background counts as white if >= 80% of edge pixels are white.

    80% (not 100%) because products often touch the edge of the photo.
    """
    return white_border_fraction(img) >= threshold
