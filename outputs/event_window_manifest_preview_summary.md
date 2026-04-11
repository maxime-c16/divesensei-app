# Event Window Manifest Preview

- preview rows: `341`
- sessions used: `3`
- primary anchor: `proposal_centered`
- backup anchor: `earliest_strong_peak_in_local_cluster`

## Sessions

| source root | source session | session type | provenance | reviewed candidates | false negatives |
|---|---|---|---|---:|---:|
| `/Users/mcauchy/divesensei-app/outputs/evaluation_insep_15min_validated` | `evaluation_insep_15min_validated` | `springboard` | `direct_review` | 133 | 16 |
| `/Users/mcauchy/divesensei-app/outputs/evaluation_champigny_20260406-labelling` | `evaluation_champigny_20260406-labelling` | `springboard` | `session_type_inferred` | 111 | 3 |
| `/Users/mcauchy/divesensei-app/outputs/evaluation_insep_quick_9015_20260409_ui` | `evaluation_insep_quick_9015_20260409_ui` | `platform` | `session_type_inferred` | 71 | 7 |

## Counts

- event labels: `{"noise_or_other": 52, "platform_dive": 43, "springboard_dive": 105, "springboard_rebound_only": 141}`
- label provenance: `{"session_type_inferred": 148, "subtype_mapped": 190, "uncertain": 3}`
- session type: `{"platform": 78, "springboard": 263}`
- session type provenance: `{"direct_review": 149, "session_type_inferred": 192}`

## Notes

- legacy candidate labels and detector scores are preserved as metadata
- no legacy label semantics were rewritten in place
- all event labels in this preview are provenance-tagged
