# Event Reviewed Manifest Preview

- session: `evaluation_insep_15min_validated`
- rows: `149`
- reviewed rows: `134`
- missing final event labels: `15`

## Counts

- final labels: `{"noise_or_other": 20, "springboard_dive": 45, "springboard_rebound_only": 69}`
- legacy labels: `{"dive": 45, "false_negative": 16, "non_dive": 88}`
- suggestion agreement: `{"agree": 132, "disagree": 2}`

## Example Reviewed Rows

- `det-0001` -> final `springboard_rebound_only` / suggestion `springboard_rebound_only`
- `det-0002` -> final `springboard_rebound_only` / suggestion `springboard_rebound_only`
- `det-0003` -> final `springboard_dive` / suggestion `springboard_dive`
- `det-0004` -> final `springboard_dive` / suggestion `springboard_dive`
- `det-0005` -> final `springboard_rebound_only` / suggestion `springboard_rebound_only`

## Notes

- final event labels are human-reviewed values from the desktop review store
- legacy detector decisions and suggestion fields are preserved
- missing rows remain visible for dataset completeness tracking
