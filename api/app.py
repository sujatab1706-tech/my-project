"""FastAPI app: upload a product photo, get compliance flags back.

Run locally:
  uvicorn api.app:app --reload
Then open http://127.0.0.1:8000/docs to try it in the browser.
"""
import csv
import io
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from inference import Predictor  # noqa: E402

MODEL_PATH = os.getenv("MODEL_PATH", "models/cnn_mobilenet.keras")
THRESHOLDS_PATH = os.getenv("THRESHOLDS_PATH", "models/thresholds.json")
LOG_PATH = Path(os.getenv("LOG_PATH", "logs/predictions.csv"))
MAX_BYTES = 10 * 1024 * 1024  # 10 MB

LOG_FIELDS = ["timestamp", "filename", "tlgw_prob", "nwb_prob", "tlgw_flag", "nwb_flag", "nwb_cnn_flag",
              "nwb_rule_flag", "width", "height", "brightness", "border_white_fraction", "latency_ms"]

app = FastAPI(title="Amazon Image Compliance Checker",
              description="Flags TLGW (text/logo/graphic/watermark) and NWB (non-white background) in product photos.")
predictor = Predictor(MODEL_PATH, THRESHOLDS_PATH)


def log_prediction(filename: str, result: dict) -> None:
    """Append one row per prediction. Monitoring reads this file later.
    We store numbers only, never the image itself."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not LOG_PATH.exists()
    with LOG_PATH.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "filename": filename,
            "tlgw_prob": result["tlgw"]["probability"],
            "nwb_prob": result["nwb"]["probability"],
            "tlgw_flag": int(result["tlgw"]["flag"]),
            "nwb_flag": int(result["nwb"]["flag"]),
            "nwb_cnn_flag": int(result["nwb"]["cnn_flag"]),
            "nwb_rule_flag": int(result["nwb"]["rule_flag"]),
            **result["image"],
            "latency_ms": result["latency_ms"],
        })


@app.get("/health")
def health():
    return {"status": "ok", "model": predictor.model_path, "thresholds": predictor.thresholds}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image too large (max 10 MB)")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Could not read the file as an image")
    result = predictor.predict(img)
    log_prediction(file.filename or "upload", result)
    return {"filename": file.filename, **result}


@app.get("/stats")
def stats():
    """Quick live view: how many predictions so far, and how often each flag fires."""
    if not LOG_PATH.exists():
        return {"predictions": 0}
    with LOG_PATH.open() as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    return {
        "predictions": n,
        "tlgw_flag_rate": round(sum(int(r["tlgw_flag"]) for r in rows) / n, 3),
        "nwb_flag_rate": round(sum(int(r["nwb_flag"]) for r in rows) / n, 3),
        "avg_latency_ms": round(sum(float(r["latency_ms"]) for r in rows) / n, 1),
    }
