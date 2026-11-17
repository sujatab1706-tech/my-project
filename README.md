# Amazon Image Compliance Checker

A beginner, end-to-end ML project: look at an Amazon product photo and flag whether it breaks listing image rules.

| Label | Meaning | Status |
|---|---|---|
| **TLGW** | Text, logo, graphic or watermark added on the main image | Phase 1 |
| **NWB** | Non-white background (main images need pure white, RGB 255,255,255) | Phase 1 |
| **Brand infringement** | A brand logo the seller isn't allowed to use | Phase 2 |
| **Prurience** | Sexually suggestive or adult content | Phase 2 |

One photo can break several rules at once, so each image gets a separate yes/no per label. This is **multi-label classification**.

Everything runs free, on Google Colab or a laptop. No AWS account is needed.

## Plan

1. **Data** ✅: clean photos from [Amazon Berkeley Objects](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) (CC BY-NC 4.0), with rule violations added by code so every label is known.
2. **Models** ✅: simple rules and OCR as baselines, then a MobileNetV2 transfer-learning CNN with one output per label.
3. **Evaluation** ✅: precision, recall, F1 and a confusion matrix per label, plus a decision threshold tuned per label.
4. **Deployment and monitoring** ✅: FastAPI endpoint, Docker, GitHub Actions tests, Evidently to watch how often each flag fires.

## Step 1: build the dataset

Open `notebooks/01_build_dataset.ipynb` in Colab, or run:

```bash
pip install -r requirements.txt
python src/download_abo.py --n 3000 --out data/raw         # clean white-background photos
python src/make_violations.py --raw data/raw --out data/processed
```

Output: `data/processed/images/*.jpg` (224×224) and `data/processed/labels.csv`:

| column | meaning |
|---|---|
| `file` | image file name |
| `source_id` | original ABO photo; the clean and broken versions share it |
| `split` | train / val / test, split by `source_id` to avoid data leakage |
| `tlgw`, `nwb` | labels (1 = breaks the rule) |
| `detail` | what was done, e.g. `bg_gradient+watermark` |

`data/` is in `.gitignore`. Images are rebuilt from the scripts and never committed.

## Step 2: train the models

Open `notebooks/02_train_models.ipynb` in Colab (turn on the free GPU), or run:

```bash
# OCR needs the Tesseract program: apt-get install tesseract-ocr  (Mac: brew install tesseract)
python src/baselines.py --data data/processed --out results                     # rule + OCR, no training
python src/train_cnn.py --model small --epochs 15 --data data/processed --out results
python src/train_cnn.py --model mobilenet --epochs 10 --data data/processed --out results
```

Test-set accuracy from one run (3,626 images; your numbers will vary a little):

| Label | Always "no" | Rule / OCR | Small CNN | MobileNetV2 |
|---|---|---|---|---|
| TLGW | 0.672 | 0.677 (OCR) | 0.662 | **0.945** |
| NWB | 0.677 | **0.987** (rule) | 0.974 | 0.963 |

"Always no" never flags anything, so any useful model must beat it. For NWB a 5-line rule wins; for TLGW only transfer learning works well.

Each run saves the model (`results/cnn_*.keras`), its learning curve (`results/history_*.csv`) and per-image probabilities
(`results/*_predictions.csv`) for Step 3. `results/` is git-ignored.

## Step 3: evaluate properly

Open `notebooks/03_evaluate.ipynb` (outputs are saved, so you can read it on GitHub). It covers confusion matrices, precision/recall/F1 per label,
choosing thresholds on the validation set (saved to `results/thresholds.json`), multi-label scores, a **harder test set** and Grad-CAM heatmaps.

```bash
python src/make_hard_test.py --processed data/processed --out data/hard_test   # subtle violations on unseen products
python src/predict.py --data data/hard_test --models results --out results/hard
```

MobileNetV2, test set vs hard test set (best-F1 thresholds):

| Label | Set | Precision | Recall | F1 |
|---|---|---|---|---|
| TLGW | normal test | 0.92 | 0.91 | 0.92 |
| TLGW | hard test | 0.96 | **0.74** | 0.84 |
| NWB | normal test | 0.94 | 0.97 | 0.96 |
| NWB | hard test | 0.85 | **0.34** | 0.49 |

On the hard set the CNN misses most off-white backgrounds and tiny corner text, because it never saw them in training.
Accuracy still looked fine (0.87 for NWB), and only recall showed the problem. Next fixes: add these cases to the training data, combine
the CNN with a stricter white-border rule (`tol=6` catches 98% of off-white backgrounds), then confirm on a fresh hold-out set.

## Step 4: deploy and monitor

Open `notebooks/04_deploy_and_monitor.ipynb` (runs on Colab, no Docker needed) or run it for real:

```bash
pip install -r requirements-api.txt
uvicorn api.app:app --reload                 # open http://127.0.0.1:8000/docs and try POST /predict
# or with Docker
docker build -t image-compliance .
docker run -p 8000:8000 image-compliance
```

`POST /predict` with a photo returns, for example (an off-white background the CNN misses but the rule catches):

```json
{"tlgw": {"flag": false, "probability": 0.0241, "threshold": 0.4722},
 "nwb":  {"flag": true, "probability": 0.02, "threshold": 0.1476, "cnn_flag": false, "rule_flag": true},
 "compliant": false,
 "image": {"width": 224, "height": 224, "brightness": 0.887, "border_white_fraction": 0.839},
 "latency_ms": 71.8}
```

- **Decision:** TLGW = CNN ≥ tuned threshold. NWB = CNN **or** the strict white-border rule (`tol=6`), because Step 3 showed the CNN misses off-white backgrounds.
- **Model:** `models/cnn_mobilenet.keras` (MobileNetV2 from Step 2) and `models/thresholds.json` (from Step 3) are committed so the API runs out of the box.
- **Logging:** every prediction is appended to `logs/predictions.csv` (numbers only, never the photo). `GET /stats` shows live flag rates.
- **Tests and CI:** `pytest` (API tests use a tiny fake model) plus a GitHub Actions job that builds the Docker image and smoke-tests it with the real model.
- **Monitoring:** `monitoring/monitor.py` compares the log with `monitoring/reference.csv` (the model run over validation images). It checks flag rates and uses Evidently data drift on the probabilities, brightness and white-border share.

```bash
python monitoring/simulate_traffic.py --folder data/hard_test/images --limit 200   # with the API running
python monitoring/monitor.py --current logs/predictions.csv                       # prints alerts, writes an HTML report
```

With normal photos nothing fires. With the hard test set, brightness, border whiteness and NWB probability all drift, and the NWB flag rate drops from 38% to 20%.
The monitor raises an alarm without knowing any true labels.

Sample photos in `samples/` are derived from [Amazon Berkeley Objects](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) (CC BY-NC 4.0).

## Project layout

```
src/
  download_abo.py      # download clean product photos
  make_violations.py   # create labelled TLGW / NWB images
  rules.py             # hand-written baseline rules (no ML)
  baselines.py         # rule (NWB) + OCR (TLGW) predictions
  train_cnn.py         # small CNN or MobileNetV2, 2 sigmoid outputs
  evaluate.py          # confusion matrix, precision/recall/F1, thresholds, multi-label scores
  make_hard_test.py    # subtle violations: off-white, faint watermark, tiny text
  predict.py           # run saved models + baselines on any labelled folder
  gradcam.py           # heatmap of where the CNN looks
  inference.py         # Predictor used by the API: CNN + rule -> flags
api/app.py             # FastAPI: /predict, /health, /stats
models/                # trained MobileNetV2 + thresholds
monitoring/            # reference data, traffic simulator, Evidently monitor
tests/                 # pytest: API + rules
Dockerfile, requirements-api.txt, .github/workflows/ci.yml
notebooks/
  01_build_dataset.ipynb
  02_train_models.ipynb
  03_evaluate.ipynb
  04_deploy_and_monitor.ipynb
```

---
my first project for ai/ml · started on 2nd June 2026, Tuesday · Author - Sujata Behera
