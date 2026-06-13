# INVALID: formal metric aggregation bug

The formal lambda tables previously stored in this directory are invalid.

Reason: `configs/summarize_nuplan_group.py` assumed each nuPlan `aggregator_metric/*.parquet` file contained a single aggregate row and read `df.iloc[0]`. The parquet actually contained scenario-level rows, so the reported setting scores such as 0.92 were one scenario score, not the official setting aggregate. This also explains the mismatch against the FlowDrive paper scale.

The raw nuPlan parquet/log artifacts were pruned during archive compaction, so the official formal metrics cannot be recomputed from the retained archive. The smoke-gate summaries remain valid because they were computed directly from saved trajectory tensors before pruning.

Required fix: rerun the formal sweep, or restore raw `aggregator_metric/*.parquet` / runner outputs from backup, then aggregate using the correct official metric extraction.
