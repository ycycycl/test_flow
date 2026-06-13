# Quicklook

## Status

planned

## One-line purpose

lambda0.5-four-settings

## Hypothesis

A moderate risk-attention logit bias can improve closed-loop collision/TTC related behavior without excessive progress loss in post_mode=0.

## Key params

- kind: formal_group
- lambda: 0.5
- rep: None
- settings: test14-random/R, test14-random/NR, test14-hard/R, test14-hard/NR
- post_mode: 0

## Primary metric

closed_loop_weighted_average_score

## Run directory

sweeps/2026/20260612/20260612T134944Z__e26-risk-attn-flowdrive__sweep-smoke-lambda__g72a5282__h69e4f2/runs/j007__lambda0.5-four-settings__sna__he40b5e

## How to inspect

Read `metrics/summary.json`, then `result.md`; failures write `logs/999_failure.log`.
