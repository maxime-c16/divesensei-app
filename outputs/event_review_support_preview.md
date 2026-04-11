# Event Review Support Preview

- session: `evaluation_insep_15min_validated`
- session type: `springboard`
- session type provenance: `direct_review`
- rows: `149`

## Counts

- suggestion types: `{"noise_or_other": 20, "springboard_dive": 112, "springboard_rebound_only": 17}`
- suggestion confidence: `{"high": 116, "low": 33}`
- suggestion reasons: `{"board_rebound_without_delayed_entry": 17, "insufficient_context_uncertain": 33, "negative_subtype_voice_whistle": 20, "rebound_context_plus_delayed_entry": 79}`
- rebound-context hints: `97`
- delayed-entry hints: `116`
- no-rebound-context hints: `52`
- uncertain rows: `33`
- session-aware downgrades to uncertain: `0`

## Examples

### springboard_dive
- `evaluation_insep_15min_validated` @ `13.888` -> `springboard_dive` | rebound_context_plus_delayed_entry
- `evaluation_insep_15min_validated` @ `15.792` -> `springboard_dive` | insufficient_context_uncertain
- `evaluation_insep_15min_validated` @ `33.712` -> `springboard_dive` | insufficient_context_uncertain

### springboard_rebound_only
- `evaluation_insep_15min_validated` @ `13.216` -> `springboard_rebound_only` | board_rebound_without_delayed_entry
- `evaluation_insep_15min_validated` @ `41.072` -> `springboard_rebound_only` | board_rebound_without_delayed_entry
- `evaluation_insep_15min_validated` @ `57.408` -> `springboard_rebound_only` | board_rebound_without_delayed_entry

### platform_dive
- none

### noise_or_other
- `evaluation_insep_15min_validated` @ `123.872` -> `noise_or_other` | negative_subtype_voice_whistle
- `evaluation_insep_15min_validated` @ `209.088` -> `noise_or_other` | negative_subtype_voice_whistle
- `evaluation_insep_15min_validated` @ `209.840` -> `noise_or_other` | negative_subtype_voice_whistle

### uncertain
- none

## Notes

- suggestions are machine-generated hints only
- legacy candidate labels and detector scores are preserved
- human review remains the final source of truth
