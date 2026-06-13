#!/usr/bin/env bash
set -euo pipefail
ssh 24-8V100 'bash "/root/tmp_exp/e26_risk_attn_flowdrive/20260612T134944Z__e26-risk-attn-flowdrive__sweep-smoke-lambda__g72a5282__h69e4f2/configs/run_smoke.sh" j000__smoke-lambda0-consistency-hard3__sna__h6dfc27 consistency'
