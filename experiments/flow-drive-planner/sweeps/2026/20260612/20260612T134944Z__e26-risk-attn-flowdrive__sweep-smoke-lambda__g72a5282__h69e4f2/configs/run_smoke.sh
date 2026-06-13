#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then echo "usage: $0 <child_id> <consistency|effect>" >&2; exit 2; fi
CHILD_ID="$1"
MODE="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"
REMOTE_RUN="$REMOTE_SWEEP/runs/$CHILD_ID"
mkdir -p "$REMOTE_RUN/logs" "$REMOTE_RUN/metrics" "$REMOTE_RUN/artifacts" "$REMOTE_RUN/configs"
cd "$ROOT"
git rev-parse HEAD > "$REMOTE_RUN/configs/remote_git_commit.txt"
git status --short > "$REMOTE_RUN/configs/remote_git_status.txt"
python "$REMOTE_SWEEP/configs/smoke_planner_forward.py"   --mode "$MODE"   --run-dir "$REMOTE_RUN"   --root "$ROOT"   --ckpt "$CKPT_PATH"   --data-root "$DATA_ROOT/nuplan-v1.1/splits/test"   --maps-root "$MAPS_ROOT"   --device cuda 2>&1 | tee "$REMOTE_RUN/logs/020_smoke_${MODE}.log"
