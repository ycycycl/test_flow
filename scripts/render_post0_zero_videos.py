#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from flowdrive_render_utils import (
    EPS,
    METRIC_COLUMNS,
    json_default,
    render_simlog_video,
    safe_filename,
    write_json,
)


DEFAULT_POST0_SWEEP = Path(
    "/root/data1/ycl/flow_drive_planner_repro/flow_drive_planner/experiments/flow-drive-planner/"
    "sweeps/2026/20260529/20260529T121653Z__flowdrive-eval__test14-hard-random-r-nr__g06f941e__hc83baf"
)
DEFAULT_POST1_SWEEP = Path(
    "/root/data1/ycl/flow_drive_planner_repro/flow_drive_planner/experiments/flow-drive-planner/"
    "sweeps/2026/20260530/20260530T024743Z__flowdrive-refine-post1-eval__test14-hard-random-r-nr__g06f941e__hf3b3f0"
)

DEFAULT_RUNS = {
    ("post0", "test14-hard/R"): "j001__test14-hard-r__g06f941e__h43e9f2",
    ("post0", "test14-hard/NR"): "j002__test14-hard-nr__g06f941e__h07098f",
    ("post0", "test14-random/R"): "j003__test14-random-r__g06f941e__h03a507",
    ("post0", "test14-random/NR"): "j004__test14-random-nr__g06f941e__hce44f3",
    ("post1", "test14-hard/R"): "j001__test14-hard-r-post1__g06f941e__h7fecd1",
    ("post1", "test14-hard/NR"): "j002__test14-hard-nr-post1__g06f941e__hd71526",
    ("post1", "test14-random/R"): "j003__test14-random-r-post1__g06f941e__hf647c5",
    ("post1", "test14-random/NR"): "j004__test14-random-nr-post1__g06f941e__hfeb985",
}

SETTINGS = ["test14-hard/R", "test14-hard/NR", "test14-random/R", "test14-random/NR"]


def run_dir(phase: str, setting: str, args: argparse.Namespace) -> Path:
    root = args.post0_sweep if phase == "post0" else args.post1_sweep
    return root / "runs" / args.runs[f"{phase}:{setting}"]


def load_metric_table(phase: str, setting: str, args: argparse.Namespace) -> pd.DataFrame:
    path = run_dir(phase, setting, args)
    summary_path = path / "metrics" / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metric_rel = summary["metric_files"][0]
        metric_path = path / metric_rel
    else:
        matches = sorted(path.rglob("aggregator_metric/*.parquet"))
        if not matches:
            raise FileNotFoundError(f"No metric parquet found under {path}")
        metric_path = matches[-1]
    df = pd.read_parquet(metric_path)
    df = df[df["num_scenarios"].isna()].copy()
    df["phase"] = phase
    df["setting"] = setting
    df["scenario_key"] = df["log_name"].astype(str) + "|" + df["scenario"].astype(str)
    a = pd.to_numeric(df.get("ego_is_making_progress"), errors="coerce")
    b = pd.to_numeric(df.get("ego_progress_along_expert_route"), errors="coerce")
    df["progress"] = pd.concat([a, b], axis=1).mean(axis=1)
    return df


def metric_dict(row: dict[str, Any], suffix: str) -> dict[str, Any]:
    out = {}
    for key in list(METRIC_COLUMNS.values()) + ["progress"]:
        out[key] = row.get(f"{key}_{suffix}")
    out["score"] = row.get(f"score_{suffix}")
    return out


def load_cases(args: argparse.Namespace) -> pd.DataFrame:
    frames = []
    for phase in ["post0", "post1"]:
        for setting in SETTINGS:
            frames.append(load_metric_table(phase, setting, args))
    all_df = pd.concat(frames, ignore_index=True)
    post0 = all_df[all_df["phase"] == "post0"].copy()
    post1 = all_df[all_df["phase"] == "post1"].copy()
    base_cols = ["setting", "scenario_key", "log_name", "scenario", "scenario_type"]
    metric_cols = list(dict.fromkeys(list(METRIC_COLUMNS.values()) + ["progress"]))
    merged = post0[base_cols + metric_cols].merge(
        post1[base_cols + metric_cols],
        on=["setting", "scenario_key"],
        suffixes=("_post0", "_post1"),
        how="left",
    )
    zero = merged[pd.to_numeric(merged["score_post0"], errors="coerce").abs() <= EPS].copy()
    zero = zero.sort_values(["setting", "scenario_type_post0", "log_name_post0", "scenario_post0"]).reset_index(drop=True)
    zero.insert(0, "rank", range(1, len(zero) + 1))
    return zero


def index_simlogs(args: argparse.Namespace) -> dict[tuple[str, str], dict[str, Path]]:
    indices: dict[tuple[str, str], dict[str, Path]] = {}
    for phase in ["post0", "post1"]:
        for setting in SETTINGS:
            root = run_dir(phase, setting, args) / "artifacts"
            indices[(phase, setting)] = {
                path.stem.replace(".msgpack", ""): path
                for path in root.rglob("*.msgpack.xz")
            }
    return indices


def attach_msgpack_paths(cases: pd.DataFrame, indices: dict[tuple[str, str], dict[str, Path]]) -> pd.DataFrame:
    rows = []
    for _, row in cases.iterrows():
        item = row.to_dict()
        setting = str(item["setting"])
        token = str(item["scenario_post0"])
        item["post0_msgpack"] = str(indices.get(("post0", setting), {}).get(token, ""))
        item["post1_msgpack"] = str(indices.get(("post1", setting), {}).get(token, ""))
        rows.append(item)
    return pd.DataFrame(rows)


def render_one(job: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    row, config = job
    phase = str(config["phase"])
    msgpack_path = Path(row[f"{phase}_msgpack"])
    if not msgpack_path.exists():
        raise FileNotFoundError(f"Missing {phase} msgpack: {msgpack_path}")
    out_name = (
        f"{int(row['rank']):03d}_"
        f"{phase}_"
        f"{safe_filename(str(row['setting']))}_"
        f"{safe_filename(str(row['scenario_type_post0']))}_"
        f"{safe_filename(str(row['scenario_post0']))}.mp4"
    )
    title = (
        f"{row['setting']} | {row['scenario_type_post0']} | "
        f"{row['log_name_post0']} | {row['scenario_post0']}"
    )
    info = render_simlog_video(
        msgpack_path,
        Path(config["videos_dir"]) / out_name,
        metric_dict(row, phase),
        title=title,
        label=phase.upper(),
        stride=int(config["stride"]),
        fps=int(config["fps"]),
        panel_size=int(config["panel_size"]),
        bounds=float(config["bounds"]),
        offset=float(config["offset"]),
        map_cache_step=int(config["map_cache_step"]),
        quality=int(config["quality"]),
    )
    out = dict(row)
    out.update(info)
    out["render_phase"] = phase
    return out


def write_html(path: Path, records: list[dict[str, Any]], failures: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    def rel(path_str: str) -> str:
        try:
            return Path(path_str).relative_to(path.parent).as_posix()
        except Exception:
            return Path(path_str).name

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>FlowDrive Scenario Videos</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;color:#202124;background:#f7f7f7}"
        ".card{background:white;border:1px solid #ddd;border-radius:6px;padding:12px;margin:14px 0}"
        "video{max-width:100%;height:auto;border:1px solid #ddd;background:black}"
        "table{border-collapse:collapse;font-size:12px;background:white}td,th{border:1px solid #ddd;padding:4px 6px}"
        "code{background:#eee;padding:1px 4px;border-radius:3px}</style></head><body>",
        f"<h1>FlowDrive {html.escape(str(summary['render_phase']).upper())} Videos</h1>",
        f"<p>Rendered <b>{summary['rendered_videos']}</b> / {summary['requested_videos']} requested; failures: {summary['failed_videos']}.</p>",
    ]
    if records:
        cols = ["rank", "render_phase", "setting", "scenario_type_post0", "scenario_post0", "score_post0", "score_post1", "frames", "bytes"]
        parts.append(pd.DataFrame(records)[cols].to_html(index=False, escape=True))
    if failures:
        parts.append("<h2>Failures</h2><ul>")
        for fail in failures[:20]:
            parts.append(f"<li><code>{html.escape(str(fail.get('scenario_post0')))}</code>: {html.escape(str(fail.get('error')))}</li>")
        parts.append("</ul>")
    for record in records:
        src = html.escape(rel(str(record["video"])))
        title = f"{int(record['rank']):03d} {record['setting']} | {record['scenario_type_post0']} | {record['scenario_post0']}"
        parts.extend(["<div class='card'>", f"<h2>{html.escape(title)}</h2>", f"<video controls preload='metadata' src='{src}'></video>", "</div>"])
    parts.append("</body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")


def parse_run_overrides(values: list[str]) -> dict[str, str]:
    out = {f"{phase}:{setting}": run for (phase, setting), run in DEFAULT_RUNS.items()}
    for value in values:
        key, run = value.split("=", 1)
        out[key] = run
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render single-mode videos for scenarios selected by post0 zero score.")
    parser.add_argument("--post0-sweep", type=Path, default=DEFAULT_POST0_SWEEP)
    parser.add_argument("--post1-sweep", type=Path, default=DEFAULT_POST1_SWEEP)
    parser.add_argument("--run", action="append", default=[], help="Override run id, e.g. post0:test14-hard/R=j001__...")
    parser.add_argument("--phase", choices=["post0", "post1"], default="post0", help="Which simulation log to render. post0 renders only post0; post1 renders only post1.")
    parser.add_argument("--out-dir", type=Path, default=Path("post0_zero_videos"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--panel-size", type=int, default=720)
    parser.add_argument("--bounds", type=float, default=60.0)
    parser.add_argument("--offset", type=float, default=20.0)
    parser.add_argument("--map-cache-step", type=int, default=20)
    parser.add_argument("--quality", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.runs = parse_run_overrides(args.run)
    out_dir = args.out_dir.resolve()
    artifacts_dir = out_dir / "artifacts"
    videos_dir = artifacts_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    cases = attach_msgpack_paths(load_cases(args), index_simlogs(args))
    cases.to_csv(artifacts_dir / "post0_zero_cases.csv", index=False)
    render_cases = cases.head(args.limit).copy() if args.limit and args.limit > 0 else cases.copy()
    config = {
        "videos_dir": str(videos_dir),
        "stride": args.stride,
        "fps": args.fps,
        "panel_size": args.panel_size,
        "bounds": args.bounds,
        "offset": args.offset,
        "map_cache_step": args.map_cache_step,
        "quality": args.quality,
        "phase": args.phase,
    }
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    jobs = [(row.dropna().to_dict(), config) for _, row in render_cases.iterrows()]
    if args.workers <= 1:
        for job in jobs:
            try:
                record = render_one(job)
                records.append(record)
                print(f"[ok] {record['rank']} {record['video']}", flush=True)
            except Exception as exc:
                fail = dict(job[0])
                fail.update({"error": repr(exc), "traceback": traceback.format_exc(limit=12)})
                failures.append(fail)
                print(f"[fail] {job[0].get('rank')} {exc!r}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(render_one, job): job[0] for job in jobs}
            for future in as_completed(futures):
                row = futures[future]
                try:
                    record = future.result()
                    records.append(record)
                    print(f"[ok] {record['rank']} {record['video']}", flush=True)
                except Exception as exc:
                    fail = dict(row)
                    fail.update({"error": repr(exc), "traceback": traceback.format_exc(limit=12)})
                    failures.append(fail)
                    print(f"[fail] {row.get('rank')} {exc!r}", flush=True)
    records = sorted(records, key=lambda r: int(r["rank"]))
    failures = sorted(failures, key=lambda r: int(r.get("rank", 10**9)))
    summary = {
        "selected_zero_scenarios": int(len(cases)),
        "requested_videos": int(len(render_cases)),
        "rendered_videos": int(len(records)),
        "failed_videos": int(len(failures)),
        "render_phase": args.phase,
        "post0_sweep": str(args.post0_sweep),
        "post1_sweep": str(args.post1_sweep),
        "stride": args.stride,
        "fps": args.fps,
    }
    pd.DataFrame(records).to_csv(artifacts_dir / "video_records.csv", index=False)
    write_json(artifacts_dir / "video_failures.json", failures)
    write_json(artifacts_dir / "summary.json", summary)
    write_html(artifacts_dir / "index.html", records, failures, summary)
    print(json.dumps(summary, indent=2, default=json_default), flush=True)


if __name__ == "__main__":
    main()
