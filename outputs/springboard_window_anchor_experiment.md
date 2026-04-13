# Springboard Window / Anchor Experiment

- sessions: `evaluation_champigny_20260406-labelling, evaluation_insep_15min_validated`
- rows used: `233`
- label counts: `{"springboard_dive": 96, "springboard_rebound_only": 137}`

## Configuration Results
- anchor `proposal_centered`, window `0.75s pre + 2.25s post` -> AUC `0.5000`, macro F1 `0.3719`, mean correct springboard_dive `0.00`
- anchor `proposal_centered`, window `1.5s pre + 2.5s post` -> AUC `0.5000`, macro F1 `0.3719`, mean correct springboard_dive `0.00`
- anchor `proposal_centered`, window `0.75s pre + 3.25s post` -> AUC `0.5000`, macro F1 `0.3719`, mean correct springboard_dive `0.00`
- anchor `proposal_centered`, window `1.0s pre + 3.0s post` -> AUC `0.5000`, macro F1 `0.3719`, mean correct springboard_dive `0.00`
- anchor `earliest_strong_peak_local_cluster`, window `0.75s pre + 2.25s post` -> AUC `0.7486`, macro F1 `0.7583`, mean correct springboard_dive `24.50`
- anchor `earliest_strong_peak_local_cluster`, window `1.5s pre + 2.5s post` -> AUC `0.7486`, macro F1 `0.7583`, mean correct springboard_dive `24.50`
- anchor `earliest_strong_peak_local_cluster`, window `0.75s pre + 3.25s post` -> AUC `0.7486`, macro F1 `0.7583`, mean correct springboard_dive `24.50`
- anchor `earliest_strong_peak_local_cluster`, window `1.0s pre + 3.0s post` -> AUC `0.7486`, macro F1 `0.7583`, mean correct springboard_dive `24.50`
- anchor `delayed_entry_centered_proxy`, window `0.75s pre + 2.25s post` -> AUC `0.6787`, macro F1 `0.4859`, mean correct springboard_dive `18.00`
- anchor `delayed_entry_centered_proxy`, window `1.5s pre + 2.5s post` -> AUC `0.6787`, macro F1 `0.4859`, mean correct springboard_dive `18.00`
- anchor `delayed_entry_centered_proxy`, window `0.75s pre + 3.25s post` -> AUC `0.6787`, macro F1 `0.4859`, mean correct springboard_dive `18.00`
- anchor `delayed_entry_centered_proxy`, window `1.0s pre + 3.0s post` -> AUC `0.6787`, macro F1 `0.4859`, mean correct springboard_dive `18.00`

## Best Configurations

- best by AUC: `earliest_strong_peak_local_cluster` / `{'pre': 0.75, 'post': 2.25, 'total': 3.0}` -> AUC `0.7486`
- best by springboard_dive recovery: `earliest_strong_peak_local_cluster` / `{'pre': 0.75, 'post': 2.25, 'total': 3.0}` -> mean correct springboard_dive `24.50`

## Decision

A. Springboard becomes learnable with better window/anchor design

## Interpretation

- The experiment is springboard-only and keeps the same simple baseline family.
- Anchor/window changes are judged by whether they recover any reliable springboard_dive predictions.
- If the best configuration still fails to recover springboard_dive, the bottleneck is not just representation.
