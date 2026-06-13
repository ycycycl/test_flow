#!/usr/bin/env bash
set -euo pipefail
REMOTE_SWEEP="/root/tmp_exp/e26_risk_attn_flowdrive/20260612T134944Z__e26-risk-attn-flowdrive__sweep-smoke-lambda__g72a5282__h69e4f2"
ssh 24-8V100 "bash '$REMOTE_SWEEP/configs/run_smoke.sh' j000__smoke-lambda0-consistency-hard3__sna__h6dfc27 consistency"
ssh 24-8V100 "bash '$REMOTE_SWEEP/configs/run_smoke.sh' j001__smoke-lambda5-effect-close-front__sna__hf0e4d8 effect"
ssh 24-8V100 "nohup bash '$REMOTE_SWEEP/configs/run_all_formal_serial.sh' > '$REMOTE_SWEEP/logs/030_formal_driver.log' 2>&1 &"
