#!/usr/bin/env python
import argparse
import glob
import json
from pathlib import Path

import pandas as pd


def scalarize(v):
    try:
        if hasattr(v, "item"):
            return v.item()
    except Exception:
        pass
    return v


def read_aggregator(path):
    df = pd.read_parquet(path)
    result = {}
    # NuPlan aggregator is usually one row with metric columns.
    if len(df) > 0:
        row = df.iloc[0].to_dict()
        for k, v in row.items():
            result[str(k)] = scalarize(v)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    agg_paths = sorted(glob.glob(str(run_dir / "artifacts" / "nuplan_exp" / "**" / "aggregator_metric" / "*.parquet"), recursive=True))
    report_paths = sorted(glob.glob(str(run_dir / "artifacts" / "nuplan_exp" / "**" / "runner_report.parquet"), recursive=True))
    settings = []
    for p in agg_paths:
        metrics = read_aggregator(p)
        primary = None
        for key in ["score", "weighted_average", "closed_loop_weighted_average_score"]:
            if key in metrics:
                primary = metrics[key]
                break
        if primary is None:
            numeric = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            primary = next(iter(numeric.values())) if numeric else None
        settings.append({"aggregator_metric_path": p, "primary_metric_value": primary, "metrics": metrics})
    num_success = None
    num_failed = None
    if report_paths:
        reports = [pd.read_parquet(p) for p in report_paths]
        rep = pd.concat(reports, ignore_index=True)
        if "succeeded" in rep.columns:
            num_success = int(rep["succeeded"].sum())
            num_failed = int((~rep["succeeded"].astype(bool)).sum())
    values = [s["primary_metric_value"] for s in settings if isinstance(s["primary_metric_value"], (int, float))]
    summary = {
        "status": "completed" if values else "failed",
        "primary_metric": "closed_loop_weighted_average_score_mean_over_settings",
        "primary_metric_value": sum(values) / len(values) if values else None,
        "higher_is_better": True,
        "num_settings_found": len(settings),
        "num_runner_success": num_success,
        "num_runner_failed": num_failed,
        "settings": settings,
        "runner_report_paths": report_paths,
    }
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    with (run_dir / "metrics" / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    if not values:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
