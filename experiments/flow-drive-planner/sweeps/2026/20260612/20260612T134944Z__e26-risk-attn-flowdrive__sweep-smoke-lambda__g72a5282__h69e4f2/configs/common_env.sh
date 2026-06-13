#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/data1/ycl/Experiments/FlowDrive/E26-risk-based-attn-flowdrive
CONDA_SH=/root/data1/link_anaconda/anaconda3/etc/profile.d/conda.sh
ENV_PREFIX=/root/data1/ycl/conda_envs/flowdrive
NUPLAN_DEVKIT_ROOT=/root/data1/ycl/flow_drive_planner_repro/nuplan-devkit
INTERPLAN_DEVKIT_ROOT=/root/data1/ycl/flow_drive_planner_repro/interPlan
DATA_ROOT=/root/data1/ycl/Datasets/nuplan/dataset
MAPS_ROOT=/root/data1/ycl/Datasets/nuplan/dataset/maps
CKPT_PATH="$ROOT/flow_drive/checkpoint/flow_drive_model.pth"
REMOTE_SWEEP=/root/tmp_exp/e26_risk_attn_flowdrive/20260612T134944Z__e26-risk-attn-flowdrive__sweep-smoke-lambda__g72a5282__h69e4f2
source "$CONDA_SH"
conda activate "$ENV_PREFIX"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUPLAN_DEVKIT_ROOT INTERPLAN_DEVKIT_ROOT
export NUPLAN_DATA_ROOT="$DATA_ROOT"
export NUPLAN_MAPS_ROOT="$MAPS_ROOT"
export PYTHONPATH="$ROOT:$NUPLAN_DEVKIT_ROOT:$INTERPLAN_DEVKIT_ROOT:${PYTHONPATH:-}"
mkdir -p "$REMOTE_SWEEP/logs" "$REMOTE_SWEEP/metrics"
