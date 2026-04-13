# Event Reviewed Manifest Preview

- session: `evaluation_insep_quick_9015_20260409_ui`
- rows: `78`
- reviewed rows: `71`
- missing final event labels: `7`

## Counts

- final labels: `{"noise_or_other": 30, "platform_dive": 36, "springboard_rebound_only": 5}`
- legacy labels: `{"dive": 37, "false_negative": 7, "non_dive": 34}`
- suggestion agreement: `{"agree": 70, "disagree": 1}`

## Example Reviewed Rows

- `det-0001` -> final `noise_or_other` / suggestion `noise_or_other`
- `det-0002` -> final `platform_dive` / suggestion `platform_dive`
- `det-0003` -> final `platform_dive` / suggestion `platform_dive`
- `det-0004` -> final `platform_dive` / suggestion `platform_dive`
- `det-0005` -> final `noise_or_other` / suggestion `noise_or_other`

## Notes

- final event labels are human-reviewed values from the desktop review store
- legacy detector decisions and suggestion fields are preserved
- missing rows remain visible for dataset completeness tracking
