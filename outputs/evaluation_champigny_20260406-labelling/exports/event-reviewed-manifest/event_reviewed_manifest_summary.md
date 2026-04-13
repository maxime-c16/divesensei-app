# Event Reviewed Manifest Preview

- session: `evaluation_champigny_20260406-labelling`
- rows: `114`
- reviewed rows: `111`
- missing final event labels: `3`

## Counts

- final labels: `{"noise_or_other": 2, "platform_dive": 8, "springboard_dive": 33, "springboard_rebound_only": 68}`
- legacy labels: `{"dive": 41, "false_negative": 3, "non_dive": 70}`
- suggestion agreement: `{"agree": 111, "disagree": 0}`

## Example Reviewed Rows

- `det-0001` -> final `noise_or_other` / suggestion `noise_or_other`
- `det-0002` -> final `platform_dive` / suggestion `platform_dive`
- `det-0003` -> final `platform_dive` / suggestion `platform_dive`
- `det-0004` -> final `platform_dive` / suggestion `platform_dive`
- `det-0005` -> final `platform_dive` / suggestion `platform_dive`

## Notes

- final event labels are human-reviewed values from the desktop review store
- legacy detector decisions and suggestion fields are preserved
- missing rows remain visible for dataset completeness tracking
