#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/run_formal_group.sh" j002__lambda0-rep1-four-settings__sna__h304e84 0.0 1
bash "$SCRIPT_DIR/run_formal_group.sh" j003__lambda0-rep2-four-settings__sna__h7ed98d 0.0 2
bash "$SCRIPT_DIR/run_formal_group.sh" j004__lambda0-rep3-four-settings__sna__h6e11fe 0.0 3
bash "$SCRIPT_DIR/run_formal_group.sh" j005__lambda0.2-four-settings__sna__h3a951c 0.2 na
bash "$SCRIPT_DIR/run_formal_group.sh" j006__lambda0.3-four-settings__sna__ha2672f 0.3 na
bash "$SCRIPT_DIR/run_formal_group.sh" j007__lambda0.5-four-settings__sna__he40b5e 0.5 na
bash "$SCRIPT_DIR/run_formal_group.sh" j008__lambda0.6-four-settings__sna__he6e2fc 0.6 na
bash "$SCRIPT_DIR/run_formal_group.sh" j009__lambda0.8-four-settings__sna__ha8333b 0.8 na
bash "$SCRIPT_DIR/run_formal_group.sh" j010__lambda1-four-settings__sna__h051bd4 1.0 na
bash "$SCRIPT_DIR/run_formal_group.sh" j011__lambda2-four-settings__sna__h30e4c6 2.0 na
