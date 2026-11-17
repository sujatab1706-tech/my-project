"""Compare live predictions (the API log) with the reference data.

Two kinds of checks:
  1. FLAG RATES: did the share of images flagged TLGW / NWB change a lot?
     A sudden jump can mean sellers changed behaviour... or the model broke.
  2. DATA DRIFT (Evidently): do the incoming photos and the model's scores
     look statistically different from the reference? For example, photos
     getting darker, or probabilities piling up around 0.5 (the model is unsure).

We can't measure accuracy live because nobody labels incoming photos.
Watching flag rates and drift is how you notice problems anyway.

Usage:
  python monitoring/monitor.py --reference monitoring/reference.csv --current logs/predictions.csv
"""
import argparse
import sys

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

COLUMNS = ["tlgw_prob", "nwb_prob", "brightness", "border_white_fraction"]
FLAGS = ["tlgw_flag", "nwb_flag"]


def flag_rate_check(ref, cur, max_change=0.10):
    rows, alerts = [], []
    for col in FLAGS:
        r, c = ref[col].mean(), cur[col].mean()
        rows.append({"flag": col, "reference_rate": round(r, 3), "current_rate": round(c, 3), "change": round(c - r, 3)})
        if abs(c - r) > max_change:
            alerts.append(f"{col} rate moved from {r:.0%} to {c:.0%}")
    return pd.DataFrame(rows), alerts


def drift_report(ref, cur, html_path=None):
    """Run Evidently's data drift preset on each column and save an HTML report.

    For every column Evidently picks a statistical test (e.g. Kolmogorov-Smirnov)
    that asks: could these two sets of numbers come from the same distribution?
      p_value methods:  drift if p_value < threshold (usually 0.05)
      distance methods: drift if distance >= threshold
    """
    result = Report([DataDriftPreset()]).run(reference_data=ref[COLUMNS], current_data=cur[COLUMNS])
    if html_path:
        result.save_html(html_path)
    rows = []
    for m in result.dict()["metrics"]:
        cfg = m.get("config", {})
        if not cfg.get("type", "").endswith("ValueDrift"):
            continue
        score, thr, method = float(m["value"]), cfg["threshold"], cfg["method"]
        drifted = score < thr if "p_value" in method else score >= thr
        rows.append({"column": cfg["column"], "method": method, "score": round(score, 4),
                     "threshold": thr, "drift": bool(drifted)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="monitoring/reference.csv")
    ap.add_argument("--current", default="logs/predictions.csv")
    ap.add_argument("--html", default="monitoring/drift_report.html")
    ap.add_argument("--min-rows", type=int, default=30)
    args = ap.parse_args()

    ref, cur = pd.read_csv(args.reference), pd.read_csv(args.current)
    if len(cur) < args.min_rows:
        sys.exit(f"Only {len(cur)} predictions logged; need at least {args.min_rows} for a fair comparison.")

    rates, alerts = flag_rate_check(ref, cur)
    print("Flag rates\n", rates.to_string(index=False), "\n")
    drift = drift_report(ref, cur, args.html)
    print("Data drift (Evidently)\n", drift.to_string(index=False))
    for r in drift[drift.drift].itertuples():
        alerts.append(f"drift detected in {r.column}")
    print(f"\nHTML report: {args.html}")
    print("\nALERTS:\n  " + "\n  ".join(alerts) if alerts else "\nNo alerts.")


if __name__ == "__main__":
    main()
