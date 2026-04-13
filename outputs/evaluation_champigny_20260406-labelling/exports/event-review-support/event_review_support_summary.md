# Event Review Support Preview

- session: `evaluation_champigny_20260406-labelling`
- session type: `springboard`
- session type provenance: `direct_review`
- rows: `114`

## Counts

- suggestion types: `{"noise_or_other": 2, "platform_dive": 8, "springboard_dive": 36, "springboard_rebound_only": 68}`
- suggestion confidence: `{"high": 59, "low": 16, "medium": 39}`
- suggestion reasons: `{"board_rebound_with_delayed_entry_context": 39, "board_rebound_without_delayed_entry": 29, "negative_subtype_handling_noise": 1, "negative_subtype_non_dive_splash": 1, "platform_session_dive_without_rebound_context": 8, "rebound_context_plus_delayed_entry": 20, "springboard_dive_without_rebound_context": 16}`
- rebound-context hints: `88`
- delayed-entry hints: `83`
- no-rebound-context hints: `26`
- uncertain rows: `0`
- session-aware downgrades to uncertain: `0`

## Examples

### springboard_dive
- `evaluation_champigny_20260406-labelling` @ `632.274` -> `springboard_dive` | springboard_dive_without_rebound_context
- `evaluation_champigny_20260406-labelling` @ `715.106` -> `springboard_dive` | springboard_dive_without_rebound_context
- `evaluation_champigny_20260406-labelling` @ `825.826` -> `springboard_dive` | rebound_context_plus_delayed_entry

### springboard_rebound_only
- `evaluation_champigny_20260406-labelling` @ `807.378` -> `springboard_rebound_only` | board_rebound_without_delayed_entry
- `evaluation_champigny_20260406-labelling` @ `809.618` -> `springboard_rebound_only` | board_rebound_without_delayed_entry
- `evaluation_champigny_20260406-labelling` @ `810.194` -> `springboard_rebound_only` | board_rebound_without_delayed_entry

### platform_dive
- `evaluation_champigny_20260406-labelling` @ `70.416` -> `platform_dive` | platform_session_dive_without_rebound_context
- `evaluation_champigny_20260406-labelling` @ `167.264` -> `platform_dive` | platform_session_dive_without_rebound_context
- `evaluation_champigny_20260406-labelling` @ `254.464` -> `platform_dive` | platform_session_dive_without_rebound_context

### noise_or_other
- `evaluation_champigny_20260406-labelling` @ `2.498` -> `noise_or_other` | negative_subtype_handling_noise
- `evaluation_champigny_20260406-labelling` @ `516.370` -> `noise_or_other` | negative_subtype_non_dive_splash

### uncertain
- none

## Notes

- suggestions are machine-generated hints only
- legacy candidate labels and detector scores are preserved
- human review remains the final source of truth
