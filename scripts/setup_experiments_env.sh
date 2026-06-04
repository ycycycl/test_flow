#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_FLOWDRIVE_REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"

FLOWDRIVE_REPO="${FLOWDRIVE_REPO:-${DEFAULT_FLOWDRIVE_REPO}}"
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-$(cd "${FLOWDRIVE_REPO}/.." && pwd)}"
DEFAULT_NUPLAN_DEVKIT_ROOT="${EXPERIMENTS_ROOT}/nuplan-devkit"
DEFAULT_INTERPLAN_DEVKIT_ROOT="${EXPERIMENTS_ROOT}/interPlan"

if [[ ! -e "${DEFAULT_NUPLAN_DEVKIT_ROOT}" && -e /root/data1/ycl/flow_drive_planner_repro/nuplan-devkit ]]; then
  DEFAULT_NUPLAN_DEVKIT_ROOT=/root/data1/ycl/flow_drive_planner_repro/nuplan-devkit
fi
if [[ ! -e "${DEFAULT_INTERPLAN_DEVKIT_ROOT}" && -e /root/data1/ycl/flow_drive_planner_repro/interPlan ]]; then
  DEFAULT_INTERPLAN_DEVKIT_ROOT=/root/data1/ycl/flow_drive_planner_repro/interPlan
fi

export FLOWDRIVE_REPO
export NUPLAN_DEVKIT_ROOT="${NUPLAN_DEVKIT_ROOT:-${DEFAULT_NUPLAN_DEVKIT_ROOT}}"
export INTERPLAN_DEVKIT_ROOT="${INTERPLAN_DEVKIT_ROOT:-${DEFAULT_INTERPLAN_DEVKIT_ROOT}}"
export NUPLAN_DATA_ROOT="${NUPLAN_DATA_ROOT:-/root/data1/ycl/Datasets/nuplan/dataset}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/root/data1/ycl/Datasets/nuplan/dataset/maps}"

export PYTHONPATH="${FLOWDRIVE_REPO}:${NUPLAN_DEVKIT_ROOT}:${INTERPLAN_DEVKIT_ROOT}:${PYTHONPATH:-}"

if [[ "${FLOWDRIVE_ENV_QUIET:-0}" != "1" ]]; then
  echo "FLOWDRIVE_REPO=${FLOWDRIVE_REPO}"
  echo "NUPLAN_DEVKIT_ROOT=${NUPLAN_DEVKIT_ROOT}"
  echo "INTERPLAN_DEVKIT_ROOT=${INTERPLAN_DEVKIT_ROOT}"
  echo "NUPLAN_DATA_ROOT=${NUPLAN_DATA_ROOT}"
  echo "NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT}"
fi
