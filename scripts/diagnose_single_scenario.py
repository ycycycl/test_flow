#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from flowdrive_render_utils import (
    METRIC_COLUMNS,
    float_or_none,
    json_default,
    render_simlog_video,
    safe_filename,
    write_json,
)


KEY_METRICS = [
    "score",
    "no_ego_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
    "ego_is_making_progress",
    "ego_progress_along_expert_route",
    "ego_is_comfortable",
    "speed_limit_compliance",
    "driving_direction_compliance",
]


def parse_scenario_id(value: str) -> tuple[str | None, str]:
    if "|" in value:
        log_name, token = value.split("|", 1)
        return log_name.strip() or None, token.strip()
    return None, value.strip()


def auto_scenario_builder(split: str) -> str:
    return "nuplan" if "val14" in split else "nuplan_challenge"


def list_value(values: list[str]) -> str:
    return "[" + ",".join(values) + "]"


def build_run_command(args: argparse.Namespace, run_dir: Path, token: str, log_name: str | None, post_mode: int) -> list[str]:
    nuplan_root = Path(args.nuplan_devkit_root or os.environ.get("NUPLAN_DEVKIT_ROOT", ""))
    if not nuplan_root:
        raise ValueError("Set --nuplan-devkit-root or NUPLAN_DEVKIT_ROOT")
    run_script = nuplan_root / "nuplan" / "planning" / "script" / "run_simulation.py"
    if not args.dry_run and not run_script.exists():
        raise FileNotFoundError(f"Missing nuPlan run_simulation.py: {run_script}")

    output_dir = run_dir / "nuplan_eval"
    render_dir = run_dir / "flowdrive_render"
    diagnostic_dir = run_dir / "diagnostics"
    scenario_builder = args.scenario_builder or auto_scenario_builder(args.split)
    command = [
        sys.executable,
        str(run_script),
        f"+simulation={args.challenge}",
        "planner=flow_drive",
        f"planner.flow_drive.ckpt_path={args.ckpt_path}",
        f"planner.flow_drive.mlflow_exp_name={args.mlflow_exp_name}",
        f"planner.flow_drive.load_run_name={args.load_run_name}",
        f"planner.flow_drive.load_epoch={args.load_epoch}",
        f"planner.flow_drive.post_mode={post_mode}",
        "planner.flow_drive.render=true",
        f"planner.flow_drive.video_dir={render_dir}",
        f"planner.flow_drive.diagnostic_dir={diagnostic_dir}",
        f"scenario_builder={scenario_builder}",
        f"scenario_filter={args.split}",
        f"scenario_filter.scenario_tokens={list_value([token])}",
        "scenario_filter.limit_total_scenarios=1",
        "scenario_filter.shuffle=false",
        f"experiment_uid=flow_drive/{args.split}/single_{safe_filename(token)}_post{post_mode}",
        f"output_dir={output_dir}",
        f"verbose={str(args.verbose).lower()}",
        f"worker={args.worker}",
        f"worker.threads_per_node={args.threads_per_node}",
        "distributed_mode=SINGLE_NODE",
        f"number_of_gpus_allocated_per_simulation={args.gpu_per_simulation}",
        "enable_simulation_progress_bar=true",
        "hydra.searchpath=[pkg://flow_drive.config.scenario_filter,pkg://flow_drive.config,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]",
    ]
    if args.data_root:
        command.append(f"scenario_builder.data_root={args.data_root}")
    if log_name:
        command.append(f"scenario_filter.log_names={list_value([log_name])}")
    if args.scenario_type:
        command.append(f"scenario_filter.scenario_types={list_value([args.scenario_type])}")
    command.extend(args.extra_override or [])
    return command


def find_one(pattern_root: Path, pattern: str) -> Path | None:
    matches = sorted(pattern_root.rglob(pattern))
    return matches[0] if matches else None


def find_latest_aggregator(output_dir: Path) -> Path:
    files = sorted(output_dir.rglob("aggregator_metric/*.parquet"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No aggregator_metric parquet found under {output_dir}")
    return files[-1]


def normalize_progress(row: dict[str, Any]) -> None:
    a = float_or_none(row.get("ego_is_making_progress"))
    b = float_or_none(row.get("ego_progress_along_expert_route"))
    vals = [v for v in [a, b] if v is not None]
    row["progress"] = sum(vals) / len(vals) if vals else None


def load_score_detail(output_dir: Path, token: str) -> dict[str, Any]:
    aggregator_file = find_latest_aggregator(output_dir)
    agg = pd.read_parquet(aggregator_file)
    scenario_rows = agg[agg["num_scenarios"].isna()].copy()
    if token in set(scenario_rows["scenario"].astype(str)):
        row = scenario_rows[scenario_rows["scenario"].astype(str) == token].iloc[0].to_dict()
    elif len(scenario_rows) == 1:
        row = scenario_rows.iloc[0].to_dict()
    else:
        raise RuntimeError(f"Could not identify scenario row for token {token} in {aggregator_file}")
    normalize_progress(row)
    subscores = {key: row.get(key) for key in KEY_METRICS if key in row}
    subscores["progress"] = row.get("progress")
    return {
        "aggregator_file": str(aggregator_file),
        "scenario_token": str(row.get("scenario", token)),
        "log_name": row.get("log_name"),
        "scenario_type": row.get("scenario_type"),
        "planner_name": row.get("planner_name"),
        "score": row.get("score"),
        "subscores_from_aggregator": subscores,
        "all_aggregator_rows": scenario_rows.to_dict(orient="records"),
    }


def collect_metric_file_details(output_dir: Path, token: str) -> dict[str, Any]:
    details: dict[str, Any] = {}
    metrics_dir = output_dir / "metrics"
    for metric_name in KEY_METRICS:
        if metric_name == "score":
            continue
        path = find_one(metrics_dir, f"{metric_name}.parquet")
        if path is None:
            continue
        try:
            df = pd.read_parquet(path)
            rows = df[df["scenario"].astype(str) == token] if "scenario" in df.columns else df
            if rows.empty and len(df) == 1:
                rows = df
            if rows.empty:
                continue
            details[metric_name] = {
                "file": str(path),
                "row": rows.iloc[0].to_dict(),
            }
        except Exception as exc:
            details[metric_name] = {"file": str(path), "error": repr(exc)}
    return details


def read_selector_diagnostics(diagnostic_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(diagnostic_dir.glob("selector_frames_*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    item["source_file"] = str(path)
                    rows.append(item)
    all_zero = [row for row in rows if row.get("all_scores_zero")]
    stop_applied = [row for row in rows if row.get("allzero_stop_applied")]
    return {
        "selector_frame_count": len(rows),
        "all_zero_score_frames": len(all_zero),
        "allzero_stop_applied_frames": len(stop_applied),
        "all_zero_iterations": [row.get("iteration") for row in all_zero],
        "rows": rows,
    }


def read_initial_condition_diagnostics(diagnostic_dir: Path) -> dict[str, Any]:
    records = []
    for path in sorted(diagnostic_dir.glob("initial_condition_*.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            item["source_file"] = str(path)
            records.append(item)
        except Exception as exc:
            records.append({"source_file": str(path), "error": repr(exc)})
    summary = {"records": records}
    if records:
        first = records[0]
        summary.update(
            {
                "rear_axle_in_drivable_area": first.get("rear_axle_in_drivable_area"),
                "ego_footprint_intersects_drivable_area": first.get("ego_footprint_intersects_drivable_area"),
                "speed_mps": first.get("speed_mps"),
                "abnormal_initial_condition": (
                    first.get("rear_axle_in_drivable_area") is False
                    or first.get("ego_footprint_intersects_drivable_area") is False
                ),
            }
        )
    return summary


def write_selector_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["iteration", "selected_index", "all_scores_zero", "allzero_stop_applied", "selector_action", "scores"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["scores"] = json.dumps(out.get("scores", []))
            writer.writerow(out)


def print_score_detail(score_detail: dict[str, Any], selector_detail: dict[str, Any], initial_detail: dict[str, Any], post_mode: int) -> None:
    print("\n=== Score detail ===")
    print(f"scenario={score_detail.get('scenario_token')} log={score_detail.get('log_name')} type={score_detail.get('scenario_type')}")
    print(f"score={float_or_none(score_detail.get('score'))}")
    subscores = score_detail.get("subscores_from_aggregator", {})
    for label, key in [
        ("collision", "no_ego_at_fault_collisions"),
        ("offroad", "drivable_area_compliance"),
        ("TTC", "time_to_collision_within_bound"),
        ("progress_gate", "ego_is_making_progress"),
        ("progress_route", "ego_progress_along_expert_route"),
        ("progress", "progress"),
        ("comfort", "ego_is_comfortable"),
        ("speed_limit", "speed_limit_compliance"),
        ("direction", "driving_direction_compliance"),
    ]:
        print(f"{label}: {subscores.get(key)}")
    if post_mode == 1:
        print("\n=== Refine selector diagnostics ===")
        print(f"all_zero_score_frames={selector_detail.get('all_zero_score_frames', 0)}")
        print(f"allzero_stop_applied_frames={selector_detail.get('allzero_stop_applied_frames', 0)}")
        print(f"all_zero_iterations={selector_detail.get('all_zero_iterations', [])}")
    print("\n=== Initial condition diagnostics ===")
    print(f"rear_axle_in_drivable_area={initial_detail.get('rear_axle_in_drivable_area')}")
    print(f"ego_footprint_intersects_drivable_area={initial_detail.get('ego_footprint_intersects_drivable_area')}")
    print(f"abnormal_initial_condition={initial_detail.get('abnormal_initial_condition')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and diagnose one FlowDrive nuPlan scenario.")
    parser.add_argument("--scenario-id", required=True, help="Scenario token, or log_name|scenario_token")
    parser.add_argument("--scenario-type", default=None)
    parser.add_argument("--split", default="test14-hard")
    parser.add_argument("--challenge", default="closed_loop_reactive_agents")
    parser.add_argument("--refine", action="store_true", help="Shortcut for --post-mode 1")
    parser.add_argument("--post-mode", type=int, choices=[0, 1], default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("diagnostics"))
    parser.add_argument("--nuplan-devkit-root", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--scenario-builder", default=None)
    parser.add_argument("--ckpt-path", default=str(Path("flow_drive") / "checkpoint" / "flow_drive_model.pth"))
    parser.add_argument("--mlflow-exp-name", default="unused")
    parser.add_argument("--load-run-name", default="unused")
    parser.add_argument("--load-epoch", type=int, default=0)
    parser.add_argument("--worker", default="ray_distributed")
    parser.add_argument("--threads-per-node", type=int, default=16)
    parser.add_argument("--gpu-per-simulation", type=float, default=0.1)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--panel-size", type=int, default=720)
    parser.add_argument("--bounds", type=float, default=60.0)
    parser.add_argument("--offset", type=float, default=20.0)
    parser.add_argument("--map-cache-step", type=int, default=20)
    parser.add_argument("--quality", type=int, default=7)
    parser.add_argument("--extra-override", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_name, token = parse_scenario_id(args.scenario_id)
    post_mode = args.post_mode if args.post_mode is not None else (1 if args.refine else 0)
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}__single_{safe_filename(token)}_post{post_mode}"
    run_dir = (args.out_dir / run_id).resolve()

    command = build_run_command(args, run_dir, token, log_name, post_mode)
    if args.dry_run:
        print(" ".join(command))
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "command.json", {"command": command, "post_mode": post_mode, "scenario_token": token, "log_name": log_name})

    env = os.environ.copy()
    env["NUPLAN_EXP_ROOT"] = str(run_dir / "nuplan_eval")
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    log_path = run_dir / "run_simulation.log"
    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], env=env, stdout=log_f, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"nuPlan simulation failed with exit code {proc.returncode}; see {log_path}")

    output_dir = run_dir / "nuplan_eval"
    score_detail = load_score_detail(output_dir, token)
    metric_details = collect_metric_file_details(output_dir, token)
    selector_detail = read_selector_diagnostics(run_dir / "diagnostics")
    initial_detail = read_initial_condition_diagnostics(run_dir / "diagnostics")
    write_selector_csv(run_dir / "all_zero_selector_frames.csv", selector_detail.get("rows", []))

    msgpack = find_one(output_dir, "*.msgpack.xz")
    video_info = None
    if msgpack is not None:
        metrics = dict(score_detail["subscores_from_aggregator"])
        metrics["score"] = score_detail.get("score")
        video_info = render_simlog_video(
            msgpack,
            run_dir / "diagnostic_video.mp4",
            metrics,
            title=f"{args.split}/{args.challenge} {token} post{post_mode}",
            label=f"post{post_mode}",
            stride=args.stride,
            fps=args.fps,
            panel_size=args.panel_size,
            bounds=args.bounds,
            offset=args.offset,
            map_cache_step=args.map_cache_step,
            quality=args.quality,
        )

    summary = {
        "run_id": run_id,
        "scenario_token": token,
        "log_name": log_name or score_detail.get("log_name"),
        "scenario_type": args.scenario_type or score_detail.get("scenario_type"),
        "split": args.split,
        "challenge": args.challenge,
        "post_mode": post_mode,
        "score_detail": score_detail,
        "metric_file_details": metric_details,
        "selector_diagnostics": selector_detail,
        "initial_condition_diagnostics": initial_detail,
        "simulation_log": str(msgpack) if msgpack else None,
        "video": video_info,
    }
    write_json(run_dir / "score_detail.json", score_detail)
    write_json(run_dir / "metric_file_details.json", metric_details)
    write_json(run_dir / "summary.json", summary)
    print_score_detail(score_detail, selector_detail, initial_detail, post_mode)
    print(f"\noutputs={run_dir}")
    if video_info:
        print(f"video={video_info['video']}")


if __name__ == "__main__":
    main()
