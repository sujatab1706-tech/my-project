"""Turn one uploaded photo into compliance flags. Used by the API, tests and notebooks.

Decisions:
  tlgw = CNN probability >= tuned threshold (from Step 3)
  nwb  = CNN says NWB  OR  the strict white-border rule says NWB
         (Step 3 showed the CNN misses off-white backgrounds that the rule
          with tol=6 catches, so we combine them)
"""
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from rules import white_border_fraction

LABELS = ["tlgw", "nwb"]
IMG_SIZE = 224


def image_stats(img: Image.Image) -> dict:
    """Cheap numbers describing the photo itself. We log these so monitoring can
    notice when incoming photos start to look different from the training photos."""
    gray = np.asarray(img.convert("L"), dtype="float32")
    return {
        "width": img.width,
        "height": img.height,
        "brightness": round(float(gray.mean()) / 255, 4),
        "border_white_fraction": round(white_border_fraction(img), 4),
    }


class Predictor:
    def __init__(self, model_path="models/cnn_mobilenet.keras", thresholds_path="models/thresholds.json"):
        from tensorflow import keras  # slow import, so only when a Predictor is created

        self.model = keras.models.load_model(model_path, compile=False)
        self.thresholds = json.loads(Path(thresholds_path).read_text())
        self.model_path = str(model_path)

    def predict(self, img: Image.Image) -> dict:
        start = time.perf_counter()
        img = img.convert("RGB")
        # The CNN expects 224x224 pixels with values 0-255 (it rescales internally).
        x = np.asarray(img.resize((IMG_SIZE, IMG_SIZE)), dtype="float32")[None]
        probs = self.model.predict(x, verbose=0)[0]
        tlgw_prob, nwb_prob = float(probs[0]), float(probs[1])

        # The rule looks at the ORIGINAL photo (no resizing, no padding).
        not_white = 1 - white_border_fraction(img, tol=self.thresholds.get("nwb_rule_tol", 6))
        nwb_rule = not_white >= self.thresholds.get("nwb_rule_min_fraction", 0.2)
        nwb_cnn = nwb_prob >= self.thresholds["nwb"]

        return {
            "tlgw": {"flag": tlgw_prob >= self.thresholds["tlgw"], "probability": round(tlgw_prob, 4),
                     "threshold": self.thresholds["tlgw"]},
            "nwb": {"flag": bool(nwb_cnn or nwb_rule), "probability": round(nwb_prob, 4),
                    "threshold": self.thresholds["nwb"], "cnn_flag": bool(nwb_cnn), "rule_flag": bool(nwb_rule)},
            "compliant": not (tlgw_prob >= self.thresholds["tlgw"] or nwb_cnn or nwb_rule),
            "image": image_stats(img),
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        }
