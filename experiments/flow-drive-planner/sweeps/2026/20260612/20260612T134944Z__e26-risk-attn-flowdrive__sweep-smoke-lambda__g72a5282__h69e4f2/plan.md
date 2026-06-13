# Plan

## Goal

Validate E26 risk-based attention implementation with smoke gates, then run the formal lambda sweep over four FlowDrive settings.

## Steps

1. Confirm remote E26 repo and GPU environment.
2. Run lambda=0 consistency smoke on 2-3 test14-hard scenarios.
3. Run lambda=5 effect smoke on a close-front/lead scenario.
4. If both smoke gates pass, run formal lambda groups serially.
5. Parse official metrics and update summaries.

## Done when

- Smoke consistency reports zero trajectory difference.
- Smoke effect reports finite nonzero trajectory difference.
- Formal jobs are launched serially or completed, with per-setting summaries recorded.
