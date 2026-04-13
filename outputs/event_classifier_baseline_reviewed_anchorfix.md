# Event Classifier Baseline

Decision: `STOP_AND_ADDRESS_LABEL_GAPS`

## Dataset

- total rows: `341`
- strict rows: `255`
- practical rows: `341`
- row counts by class: `{"noise_or_other": 52, "platform_dive": 51, "springboard_dive": 96, "springboard_rebound_only": 142}`
- row counts by provenance: `{"None": 25, "human_reviewed": 316}`
- session type counts: `{"platform": 86, "springboard": 255}`
- session type provenance counts: `{"direct_review": 255, "manual_session_override": 8, "session_type_inferred": 78}`

## Split

- strategy: `leave_one_session_out`
- sessions: `evaluation_champigny_20260406-labelling, evaluation_insep_15min_validated, evaluation_insep_quick_9015_20260409_ui`
- folds: `3`

## Strict Subset

- note: direct_review session-type provenance only; useful only for the springboard contrast and not enough for a full four-class fit

## Supervision

- primary: `final_human_event_label`
- fallback: `suggested_event_label`

## Baseline

- model: `numpy logistic regression`
- features: event-window duration, anchor/proposal offsets, legacy detector scores, and provenance/session flags

## Pairwise Results

### springboard_dive_vs_springboard_rebound_only (strict)
- train/eval rows: `233`
- accuracy: `0.5879828326180258`
- AUC: `0.5`

### platform_dive_vs_noise_or_other (strict)
- train/eval rows: `22`
- accuracy: `None`
- AUC: `None`

### springboard_dive_vs_springboard_rebound_only
- mean AUC: `0.5000`
- mean accuracy: `0.7296`
- mean macro F1: `0.4146`
- mean precision: `0.3648`
- mean recall: `0.5000`

### platform_dive_vs_noise_or_other
- mean AUC: `0.6667`
- mean accuracy: `0.8630`
- mean macro F1: `0.6236`
- mean precision: `0.5982`
- mean recall: `0.6667`

## Four-Class

- macro precision: `0.2826`
- macro recall: `0.4912`
- macro F1: `0.3587`
- confusion matrix: `[[0, 96, 0, 0], [0, 137, 5, 0], [0, 0, 51, 0], [0, 22, 30, 0]]`

Per-class metrics:
- `springboard_dive`: precision `0.0000`, recall `0.0000`, f1 `0.0000`, support `96`
- `springboard_rebound_only`: precision `0.5373`, recall `0.9648`, f1 `0.6902`, support `142`
- `platform_dive`: precision `0.5930`, recall `1.0000`, f1 `0.7445`, support `51`
- `noise_or_other`: precision `0.0000`, recall `0.0000`, f1 `0.0000`, support `52`

## Strongest Confusions

- `springboard_dive_vs_springboard_rebound_only`
  - held out: `evaluation_insep_15min_validated`
  - accuracy: `0.5349`
  - confusion matrix: `[[0, 60], [0, 69]]`
- `platform_dive_vs_noise_or_other`
  - held out: `evaluation_insep_quick_9015_20260409_ui`
  - accuracy: `0.5890`
  - confusion matrix: `[[43, 0], [30, 0]]`
- `four_class`
  - macro F1: `0.3587`
  - confusion matrix: `[[0, 96, 0, 0], [0, 137, 5, 0], [0, 0, 51, 0], [0, 22, 30, 0]]`

## Decision

- `STOP_AND_ADDRESS_LABEL_GAPS`
