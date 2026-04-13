# Event Review Support Preview

- session: `evaluation_insep_15min_validated`
- session type: `springboard`
- session type provenance: `direct_review`
- rows: `149`

## Counts

- suggestion types: `{"noise_or_other": 20, "springboard_dive": 61, "springboard_rebound_only": 67, "uncertain": 1}`
- suggestion confidence: `{"high": 65, "low": 34, "medium": 50}`
- suggestion reasons: `{"board_rebound_with_delayed_entry_context": 50, "board_rebound_without_delayed_entry": 17, "insufficient_context_uncertain": 1, "negative_subtype_voice_whistle": 20, "rebound_context_plus_delayed_entry": 28, "springboard_dive_without_rebound_context": 33}`
- rebound-context hints: `96`
- delayed-entry hints: `116`
- no-rebound-context hints: `53`
- uncertain rows: `1`
- session-aware downgrades to uncertain: `0`

## Examples

### springboard_dive
- `evaluation_insep_15min_validated` @ `15.442` -> `springboard_dive` | springboard_dive_without_rebound_context
- `evaluation_insep_15min_validated` @ `33.362` -> `springboard_dive` | springboard_dive_without_rebound_context
- `evaluation_insep_15min_validated` @ `50.626` -> `springboard_dive` | springboard_dive_without_rebound_context

### springboard_rebound_only
- `evaluation_insep_15min_validated` @ `12.866` -> `springboard_rebound_only` | board_rebound_without_delayed_entry
- `evaluation_insep_15min_validated` @ `13.538` -> `springboard_rebound_only` | board_rebound_with_delayed_entry_context
- `evaluation_insep_15min_validated` @ `40.722` -> `springboard_rebound_only` | board_rebound_without_delayed_entry

### platform_dive
- none

### noise_or_other
- `evaluation_insep_15min_validated` @ `123.522` -> `noise_or_other` | negative_subtype_voice_whistle
- `evaluation_insep_15min_validated` @ `208.738` -> `noise_or_other` | negative_subtype_voice_whistle
- `evaluation_insep_15min_validated` @ `209.490` -> `noise_or_other` | negative_subtype_voice_whistle

### uncertain
- `evaluation_insep_15min_validated` @ `654.002` -> `uncertain` | insufficient_context_uncertain

## Notes

- suggestions are machine-generated hints only
- legacy candidate labels and detector scores are preserved
- human review remains the final source of truth
