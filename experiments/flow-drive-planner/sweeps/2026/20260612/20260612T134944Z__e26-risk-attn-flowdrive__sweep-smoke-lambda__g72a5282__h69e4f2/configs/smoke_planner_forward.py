#!/usr/bin/env python
import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

from flow_drive.data_process.data_processor import DataProcessor
from flow_drive.data_process.utils import convert_to_model_inputs
from flow_drive.model.flow_drive_planner import FlowDrivePlanner
from flow_drive.utils.train_utils import load_params, set_seed
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor


def get_filter_parameters(scenario_tokens):
    return (
        None,
        scenario_tokens,
        None,
        None,
        None,
        None,
        15,
        None,
        False,
        True,
        False,
        None,
        None,
        None,
    )


def load_test14_hard_tokens(root):
    path = Path(root) / "flow_drive" / "config" / "scenario_filter" / "test14-hard.yaml"
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return [str(t) for t in cfg["scenario_tokens"]]


def build_scenarios(data_root, maps_root, tokens):
    builder = NuPlanScenarioBuilder(data_root, maps_root, None, None, "nuplan-maps-v1.0")
    scenario_filter = ScenarioFilter(*get_filter_parameters(tokens))
    worker = SingleMachineParallelExecutor(use_process_pool=False)
    return builder.get_scenarios(scenario_filter, worker)


def scenario_to_inputs(processor, scenario, device):
    with tempfile.TemporaryDirectory() as tmp:
        processor._save_dir = tmp
        processor.process_scenario(scenario)
        files = list(Path(tmp).glob(f"*_{scenario.token}.npz"))
        if len(files) != 1:
            raise RuntimeError(f"expected one npz for {scenario.token}, found {files}")
        data = np.load(files[0])
        keys = [
            "neighbor_agents_past",
            "ego_current_state",
            "static_objects",
            "lanes",
            "lanes_speed_limit",
            "lanes_has_speed_limit",
            "lanes_is_route",
            "route_lanes",
            "route_lanes_speed_limit",
            "route_lanes_has_speed_limit",
        ]
        return convert_to_model_inputs({k: data[k] for k in keys if k in data}, device)


def run_once(planner, inputs, enabled, scale, seed):
    planner.params.risk_attn.enabled = enabled
    planner.params.risk_attn.logit_scale = float(scale)
    set_seed(seed)
    out = planner(inputs)
    return out.detach().cpu().numpy()


def nearest_front_distance(inputs):
    cur = inputs["neighbor_agents_past"][0, :, -1, :8].detach().cpu().numpy()
    valid = np.abs(cur).sum(axis=-1) > 0
    front = valid & (cur[:, 0] > 0.0) & (np.abs(cur[:, 1]) < 4.0)
    if front.any():
        d = np.linalg.norm(cur[front, :2], axis=-1)
        return float(d.min())
    if valid.any():
        return float(np.linalg.norm(cur[valid, :2], axis=-1).min())
    return float("inf")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["consistency", "effect"], required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--maps-root", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=520)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    for sub in ["metrics", "artifacts", "configs", "logs"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    params = load_params(str(Path(args.root) / "flow_drive" / "config" / "config.yaml"))
    params.risk_attn.enabled = False
    params.risk_attn.logit_scale = 0.0
    planner = FlowDrivePlanner(params, args.device, ckpt_path=args.ckpt)
    processor = DataProcessor(planner.params.data_processing)

    tokens = load_test14_hard_tokens(args.root)
    if args.mode == "consistency":
        selected = tokens[:3]
    else:
        # Search a small hard subset for a close lead/front object.
        candidates = build_scenarios(args.data_root, args.maps_root, tokens[:80])
        best = None
        for sc in candidates:
            if sc.scenario_type not in {"following_lane_with_lead", "stopping_with_lead", "behind_long_vehicle", "near_multiple_vehicles"}:
                continue
            inputs = scenario_to_inputs(processor, sc, args.device)
            dist = nearest_front_distance(inputs)
            if best is None or dist < best[0]:
                best = (dist, sc.token)
            if dist < 18.0:
                break
        selected = [best[1] if best is not None else tokens[0]]

    scenarios = build_scenarios(args.data_root, args.maps_root, selected)
    records = []
    max_abs_all = 0.0
    any_changed = False
    any_nan = False
    for sc in scenarios:
        inputs = scenario_to_inputs(processor, sc, args.device)
        baseline = run_once(planner, inputs, False, 0.0, args.seed)
        if args.mode == "consistency":
            candidate = run_once(planner, inputs, True, 0.0, args.seed)
        else:
            candidate = run_once(planner, inputs, True, 5.0, args.seed)
        diff = np.abs(candidate - baseline)
        max_abs = float(diff.max())
        max_abs_all = max(max_abs_all, max_abs)
        changed = bool(max_abs > 1e-8)
        finite = bool(np.isfinite(candidate).all() and np.isfinite(baseline).all())
        any_nan = any_nan or not finite
        any_changed = any_changed or changed
        np.save(run_dir / "artifacts" / f"{sc.token}_baseline.npy", baseline)
        np.save(run_dir / "artifacts" / f"{sc.token}_candidate.npy", candidate)
        records.append({
            "token": sc.token,
            "scenario_type": sc.scenario_type,
            "mode": args.mode,
            "max_abs_trajectory_diff": max_abs,
            "changed": changed,
            "finite": finite,
            "nearest_front_or_agent_distance_m": nearest_front_distance(inputs),
        })

    passed = (max_abs_all == 0.0 and not any_nan) if args.mode == "consistency" else (any_changed and not any_nan)
    summary = {
        "status": "completed" if passed else "failed",
        "mode": args.mode,
        "primary_metric": "max_abs_trajectory_diff",
        "primary_metric_value": max_abs_all,
        "higher_is_better": args.mode == "effect",
        "passed": passed,
        "records": records,
    }
    with (run_dir / "metrics" / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (run_dir / "metrics" / "metrics.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
