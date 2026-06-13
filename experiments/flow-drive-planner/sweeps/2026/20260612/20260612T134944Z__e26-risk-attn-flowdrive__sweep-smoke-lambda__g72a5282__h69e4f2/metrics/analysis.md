# E26 Risk-based Attention FlowDrive Results

## Smoke gates
- `j000__smoke-lambda0-consistency-hard3__sna__h6dfc27`: status=completed, passed=True, max_abs_trajectory_diff=0.0
- `j001__smoke-lambda5-effect-close-front__sna__hf0e4d8`: status=completed, passed=True, max_abs_trajectory_diff=59.523719787597656

## Formal lambda summary

| lambda | n | overall mean | overall std | delta vs baseline | random_R | random_NR | hard_R | hard_NR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3 | 0.895076172 | 0.000049420 | +0.000000000 | 0.904750839 | 0.823057010 | 0.922701177 | 0.929795663 |
| 0.2 | 1 | 0.895181923 | 0.000000000 | +0.000105750 | 0.904907015 | 0.823038042 | 0.922587048 | 0.930195586 |
| 0.3 | 1 | 0.895108216 | 0.000000000 | +0.000032044 | 0.904929754 | 0.822996642 | 0.922587048 | 0.929919420 |
| 0.5 | 1 | 0.894642927 | 0.000000000 | -0.000433245 | 0.904952884 | 0.822823919 | 0.922587048 | 0.928207857 |
| 0.6 | 1 | 0.894305042 | 0.000000000 | -0.000771130 | 0.904941892 | 0.822688530 | 0.922587048 | 0.927002698 |
| 0.8 | 1 | 0.893515560 | 0.000000000 | -0.001560613 | 0.904943003 | 0.822384981 | 0.922587048 | 0.924147207 |
| 1 | 1 | 0.892622022 | 0.000000000 | -0.002454150 | 0.904894588 | 0.822168919 | 0.922587048 | 0.920837533 |
| 2 | 1 | 0.662533578 | 0.000000000 | -0.232542594 | 0.906430266 | 0.821117000 | 0.922587048 | 0.000000000 |

## Baseline lambda=0 repeats

Overall baseline mean=0.895076172, std=0.000049420, values=[0.8950476399956351, 0.8950476399956351, 0.8951332371716307]

## Retained metric files

- `metrics/formal_setting_scores.csv`
- `metrics/formal_lambda_summary.csv`
- `metrics/formal_results.json`
- per-run `metrics/summary.json`
