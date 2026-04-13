# Event Review Support Preview

- session: `evaluation_insep_quick_9015_20260409_ui`
- session type: `platform`
- session type provenance: `session_type_inferred`
- rows: `78`

## Counts

- suggestion types: `{"noise_or_other": 29, "platform_dive": 44, "springboard_rebound_only": 5}`
- suggestion confidence: `{"high": 78}`
- suggestion reasons: `{"board_rebound_platform_session": 5, "negative_subtype_handling_noise": 24, "negative_subtype_voice_whistle": 5, "platform_session_dive_with_rebound_context": 2, "platform_session_dive_without_rebound_context": 42}`
- rebound-context hints: `7`
- delayed-entry hints: `50`
- no-rebound-context hints: `71`
- uncertain rows: `0`
- session-aware downgrades to uncertain: `0`

## Examples

### springboard_dive
- none

### springboard_rebound_only
- `evaluation_insep_quick_9015_20260409_ui` @ `73.216` -> `springboard_rebound_only` | board_rebound_platform_session
- `evaluation_insep_quick_9015_20260409_ui` @ `249.008` -> `springboard_rebound_only` | board_rebound_platform_session
- `evaluation_insep_quick_9015_20260409_ui` @ `249.920` -> `springboard_rebound_only` | board_rebound_platform_session

### platform_dive
- `evaluation_insep_quick_9015_20260409_ui` @ `10.096` -> `platform_dive` | platform_session_dive_without_rebound_context
- `evaluation_insep_quick_9015_20260409_ui` @ `10.688` -> `platform_dive` | platform_session_dive_without_rebound_context
- `evaluation_insep_quick_9015_20260409_ui` @ `11.728` -> `platform_dive` | platform_session_dive_without_rebound_context

### noise_or_other
- `evaluation_insep_quick_9015_20260409_ui` @ `6.112` -> `noise_or_other` | negative_subtype_handling_noise
- `evaluation_insep_quick_9015_20260409_ui` @ `14.336` -> `noise_or_other` | negative_subtype_handling_noise
- `evaluation_insep_quick_9015_20260409_ui` @ `30.560` -> `noise_or_other` | negative_subtype_handling_noise

### uncertain
- none

## Notes

- suggestions are machine-generated hints only
- legacy candidate labels and detector scores are preserved
- human review remains the final source of truth
