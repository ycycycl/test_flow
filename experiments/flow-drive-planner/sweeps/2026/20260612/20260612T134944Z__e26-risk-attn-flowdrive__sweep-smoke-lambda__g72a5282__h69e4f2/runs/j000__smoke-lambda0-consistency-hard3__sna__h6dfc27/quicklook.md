# Quicklook

## Status

planned

## One-line purpose

smoke-lambda0-consistency-hard3

## Hypothesis

With risk_attn.enabled=true and logit_scale=0.0, direct planner outputs match disabled-baseline outputs bitwise for 2-3 test14-hard scenarios.

## Key params

- kind: smoke_consistency
- lambda: 0.0
- rep: None
- settings: test14-hard initial planner forward x3
- post_mode: 0

## Primary metric

max_abs_trajectory_diff

## Run directory

sweeps/2026/20260612/20260612T134944Z__e26-risk-attn-flowdrive__sweep-smoke-lambda__g72a5282__h69e4f2/runs/j000__smoke-lambda0-consistency-hard3__sna__h6dfc27

## How to inspect

Read `metrics/summary.json`, then `result.md`; failures write `logs/999_failure.log`.
