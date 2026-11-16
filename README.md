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
3. **Evaluation**: precision, recall, F1 and a confusion matrix per label, plus a decision threshold tuned per label.
4. **Deployment and monitoring**: FastAPI endpoint, Docker, GitHub Actions tests, Evidently to watch how often each flag fires.

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

## Project layout

```
src/
  download_abo.py      # download clean product photos
  make_violations.py   # create labelled TLGW / NWB images
  rules.py             # hand-written baseline rules (no ML)
  baselines.py         # rule (NWB) + OCR (TLGW) predictions
  train_cnn.py         # small CNN or MobileNetV2, 2 sigmoid outputs
notebooks/
  01_build_dataset.ipynb
  02_train_models.ipynb
```

---
my first project for ai/ml · started on 2nd June 2026, Tuesday · Author - Sujata Behera
