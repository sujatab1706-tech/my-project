"""Evaluation helpers: everything Step 3 needs to judge a model honestly.

Key ideas (for ONE label, e.g. tlgw):
                       model says NO        model says YES
  truth NO   (fine)    True Negative (TN)   False Positive (FP)  <- false alarm
  truth YES  (broken)  False Negative (FN)  True Positive (TP)   <- caught it
                         ^ missed it!

  accuracy  = (TP + TN) / everything     "how often is it right?"
  precision = TP / (TP + FP)             "when it flags an image, how often is it right?"
  recall    = TP / (TP + FN)             "of all broken images, how many did it catch?"
  F1        = 2·P·R / (P + R)            one number that balances precision and recall

A model gives a PROBABILITY. The THRESHOLD turns it into yes/no.
  lower threshold  -> flags more -> recall up, precision down
  higher threshold -> flags less -> precision up, recall down
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)

LABELS = ["tlgw", "nwb"]


def label_metrics(y_true, y_prob, threshold=0.5) -> dict:
    """All the single-label numbers at one threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 3),
        "accuracy": (tp + tn) / len(y_true),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        # Threshold-free scores: how well the probabilities RANK broken above fine images.
        # 1.0 = perfect ranking. ROC-AUC 0.5 = coin flip; PR-AUC baseline = share of positives.
        "roc_auc": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else np.nan,
        "pr_auc": average_precision_score(y_true, y_prob) if len(set(y_true)) > 1 else np.nan,
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


def report(df: pd.DataFrame, prob_cols: dict, thresholds: dict | None = None) -> pd.DataFrame:
    """One row per label. prob_cols maps label -> column with its probability/score."""
    thresholds = thresholds or {}
    rows = []
    for label, col in prob_cols.items():
        m = label_metrics(df[label], df[col], thresholds.get(label, 0.5))
        rows.append({"label": label, **m})
    return pd.DataFrame(rows).set_index("label").round(3)


def multilabel_summary(df: pd.DataFrame, prob_cols: dict, thresholds: dict | None = None) -> dict:
    """Numbers that look at both labels together."""
    thresholds = thresholds or {}
    y_true = df[list(prob_cols)].values.astype(int)
    y_pred = np.column_stack([(df[c] >= thresholds.get(l, 0.5)).astype(int) for l, c in prob_cols.items()])
    return {
        # strict: the image only counts as right if EVERY label is right
        "exact_match": float((y_true == y_pred).all(axis=1).mean()),
        # share of single yes/no answers that are wrong
        "hamming_loss": float((y_true != y_pred).mean()),
        # macro = average of the per-label F1s (each label counts equally)
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        # micro = pool all answers together first (common labels count more)
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }


def choose_threshold(y_true, y_prob, goal="f1", min_recall=0.95) -> float:
    """Pick a threshold on the VALIDATION set (never on test: that would be cheating).

    goal="f1"     -> the threshold with the best F1 (balanced)
    goal="recall" -> the highest threshold that still catches >= min_recall of the
                     broken images (use when a miss is much worse than a false alarm)
    """
    precision, recall, thr = precision_recall_curve(y_true, y_prob)
    precision, recall = precision[:-1], recall[:-1]  # last point has no threshold
    if goal == "recall":
        ok = np.where(recall >= min_recall)[0]
        return float(thr[ok[-1]]) if len(ok) else float(thr[0])
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    return float(thr[np.argmax(f1)])
