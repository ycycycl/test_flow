# Quicklook

## Status

planned

## One-line purpose

smoke-lambda5-effect-close-front

## Hypothesis

With an exaggerated logit_scale=5.0 on a close-front/lead scenario, direct planner output should change and stay finite.

## Key params

- kind: smoke_effect
- lambda: 5.0
- rep: None
- settings: test14-hard selected close-front initial planner forward
- post_mode: 0

## Primary metric

max_abs_trajectory_diff

## Run directory

sweeps/2026/20260612/20260612T134944Z__e26-risk-attn-flowdrive__sweep-smoke-lambda__g72a5282__h69e4f2/runs/j001__smoke-lambda5-effect-close-front__sna__hf0e4d8

## How to inspect

Read `metrics/summary.json`, then `result.md`; failures write `logs/999_failure.log`.
