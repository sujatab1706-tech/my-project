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
2. **Models**: simple rules and OCR as baselines, then a MobileNetV2 transfer-learning CNN with one output per label.
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

## Project layout

```
src/
  download_abo.py      # download clean product photos
  make_violations.py   # create labelled TLGW / NWB images
  rules.py             # hand-written baseline rules (no ML)
notebooks/
  01_build_dataset.ipynb
```

---
my first project for ai/ml · started on 2nd June 2026, Tuesday · Author - Sujata Behera
