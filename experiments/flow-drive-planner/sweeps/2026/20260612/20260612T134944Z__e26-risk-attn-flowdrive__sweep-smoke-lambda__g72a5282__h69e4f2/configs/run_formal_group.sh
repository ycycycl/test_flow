#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 3 ]]; then echo "usage: $0 <child_id> <lambda> <rep>" >&2; exit 2; fi
CHILD_ID="$1"
LAMBDA="$2"
REP="$3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"
REMOTE_RUN="$REMOTE_SWEEP/runs/$CHILD_ID"
mkdir -p "$REMOTE_RUN/logs" "$REMOTE_RUN/metrics" "$REMOTE_RUN/artifacts" "$REMOTE_RUN/configs"
cd "$ROOT"
git rev-parse HEAD > "$REMOTE_RUN/configs/remote_git_commit.txt"
git status --short > "$REMOTE_RUN/configs/remote_git_status.txt"
export NUPLAN_EXP_ROOT="$REMOTE_RUN/artifacts/nuplan_exp"
mkdir -p "$NUPLAN_EXP_ROOT"

run_one() {
  local SPLIT="$1"
  local MODE="$2"
  local SIMULATION="$3"
  local SAFE_LAMBDA="${LAMBDA//./p}"
  local EXP_UID="flow_drive/e26-risk-attn/${CHILD_ID}/${SPLIT}_${MODE}"
  local JOB_NUM="${CHILD_ID:1:3}"
  local SPLIT_CODE="x"
  case "$SPLIT" in
    test14-random) SPLIT_CODE="r" ;;
    test14-hard) SPLIT_CODE="h" ;;
  esac
  # Ray appends a session/socket suffix, so the base temp path must stay short.
  export RAY_TMPDIR="/tmp/fdr${JOB_NUM}${SPLIT_CODE}${MODE}"
  export TMPDIR="/tmp/fdt${JOB_NUM}${SPLIT_CODE}${MODE}"
  mkdir -p "$RAY_TMPDIR" "$TMPDIR"
  COMMAND=(python "$NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py"     "+simulation=${SIMULATION}"     planner=flow_drive     planner.flow_drive.ckpt_path="$CKPT_PATH"     planner.flow_drive.mlflow_exp_name=unused     planner.flow_drive.load_run_name=unused     planner.flow_drive.load_epoch=0     planner.flow_drive.post_mode=0     planner.flow_drive.render=false     planner.flow_drive.video_dir=     planner.flow_drive.risk_attn.enabled=true     planner.flow_drive.risk_attn.logit_scale="$LAMBDA"     scenario_builder=nuplan_challenge     scenario_builder.data_root="$DATA_ROOT/nuplan-v1.1/splits/test"     scenario_filter="$SPLIT"     experiment_uid="$EXP_UID"     verbose=false     worker=ray_distributed     worker.threads_per_node=120     distributed_mode=SINGLE_NODE     number_of_gpus_allocated_per_simulation=0.0666667     enable_simulation_progress_bar=true     "hydra.searchpath=[pkg://flow_drive.config.scenario_filter,pkg://flow_drive.config,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]")
  printf '%q ' "${COMMAND[@]}" > "$REMOTE_RUN/configs/command_${SPLIT}_${MODE}.txt"
  printf '
' >> "$REMOTE_RUN/configs/command_${SPLIT}_${MODE}.txt"
  "${COMMAND[@]}" 2>&1 | tee "$REMOTE_RUN/logs/020_sim_${SPLIT}_${MODE}.log"
}

run_one test14-random R closed_loop_reactive_agents
run_one test14-random NR closed_loop_nonreactive_agents
run_one test14-hard R closed_loop_reactive_agents
run_one test14-hard NR closed_loop_nonreactive_agents
python "$REMOTE_SWEEP/configs/summarize_nuplan_group.py" --run-dir "$REMOTE_RUN" 2>&1 | tee "$REMOTE_RUN/logs/040_summarize.log"
