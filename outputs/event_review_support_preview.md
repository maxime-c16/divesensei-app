# Event Review Support Preview

- session: `evaluation_insep_15min_validated`
- row count: `149`
- session type: `springboard`
- session type provenance: `direct_review`

## Counts

- suggested event labels: `{"noise_or_other": 20, "platform_dive": 33, "springboard_dive": 79, "springboard_rebound_only": 17}`
- suggestion reasons: `{"board rebound with delayed-entry-like candidate": 51, "board rebound with no plausible delayed entry": 17, "dive with no rebound-like context": 33, "negative subtype mapped from voice_whistle": 20, "springboard dive with rebound context": 28}`
- rebound-context hints: `97`
- delayed-entry hints: `116`
- uncertain rows: `0`

## Examples

### springboard_dive
- `det-0002` @ `13.888` -> `board rebound with delayed-entry-like candidate` (subtype_mapped)
- `det-0008` @ `58.800` -> `board rebound with delayed-entry-like candidate` (subtype_mapped)
- `det-0009` @ `59.840` -> `springboard dive with rebound context` (session_type_inferred)
- `det-0011` @ `92.832` -> `board rebound with delayed-entry-like candidate` (subtype_mapped)
- `det-0012` @ `93.552` -> `board rebound with delayed-entry-like candidate` (subtype_mapped)

### springboard_rebound_only
- `det-0001` @ `13.216` -> `board rebound with no plausible delayed entry` (subtype_mapped)
- `det-0005` @ `41.072` -> `board rebound with no plausible delayed entry` (subtype_mapped)
- `det-0007` @ `57.408` -> `board rebound with no plausible delayed entry` (subtype_mapped)
- `det-0020` @ `132.272` -> `board rebound with no plausible delayed entry` (subtype_mapped)
- `det-0040` @ `243.920` -> `board rebound with no plausible delayed entry` (subtype_mapped)

### platform_dive
- `det-0003` @ `15.792` -> `dive with no rebound-like context` (session_type_inferred)
- `det-0004` @ `33.712` -> `dive with no rebound-like context` (session_type_inferred)
- `det-0006` @ `50.976` -> `dive with no rebound-like context` (session_type_inferred)
- `det-0010` @ `79.488` -> `dive with no rebound-like context` (session_type_inferred)
- `det-0021` @ `135.696` -> `dive with no rebound-like context` (session_type_inferred)

### noise_or_other
- `det-0019` @ `123.872` -> `negative subtype mapped from voice_whistle` (subtype_mapped)
- `det-0031` @ `209.088` -> `negative subtype mapped from voice_whistle` (subtype_mapped)
- `det-0032` @ `209.840` -> `negative subtype mapped from voice_whistle` (subtype_mapped)
- `det-0033` @ `210.816` -> `negative subtype mapped from voice_whistle` (subtype_mapped)
- `det-0060` @ `377.824` -> `negative subtype mapped from voice_whistle` (subtype_mapped)

### uncertain
- none

## Notes

- suggestions are machine-generated hints only
- legacy detector labels and scores are preserved
- human review remains the final label source
