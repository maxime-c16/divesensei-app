# Event Classifier Baseline

Decision: `STOP_AND_ADDRESS_LABEL_GAPS`

## Dataset

- total rows: `341`
- strict rows: `149`
- practical rows: `338`
- row counts by class: `{"noise_or_other": 49, "platform_dive": 43, "springboard_dive": 105, "springboard_rebound_only": 141}`
- row counts by provenance: `{"session_type_inferred": 148, "subtype_mapped": 190}`
- session type counts: `{"platform": 76, "springboard": 262}`
- session type provenance counts: `{"direct_review": 149, "session_type_inferred": 189}`

## Split

- strategy: `leave_one_session_out`
- sessions: `evaluation_champigny_20260406-labelling, evaluation_insep_15min_validated, evaluation_insep_quick_9015_20260409_ui`
- folds: `3`

## Strict Subset

- note: direct_review session-type provenance only; useful only for the springboard contrast and not enough for a full four-class fit

## Baseline

- model: `numpy logistic regression`
- features: event-window duration, anchor/proposal offsets, legacy detector scores, and provenance/session flags

## Pairwise Results

### springboard_dive_vs_springboard_rebound_only (strict)
- train/eval rows: `129`
- accuracy: `1.0`
- AUC: `1.0`

### platform_dive_vs_noise_or_other (strict)
- train/eval rows: `20`
- accuracy: `None`
- AUC: `None`

### springboard_dive_vs_springboard_rebound_only
- mean AUC: `0.8333`
- mean accuracy: `0.9940`
- mean macro F1: `0.8270`
- mean precision: `0.8286`
- mean recall: `0.8258`

### platform_dive_vs_noise_or_other
- mean AUC: `0.5014`
- mean accuracy: `0.7981`
- mean macro F1: `0.4276`
- mean precision: `0.3991`
- mean recall: `0.5000`

## Four-Class

- macro precision: `0.5759`
- macro recall: `0.5237`
- macro F1: `0.4683`
- confusion matrix: `[[105, 0, 0, 0], [0, 140, 0, 1], [43, 0, 0, 0], [0, 44, 0, 5]]`

Per-class metrics:
- `springboard_dive`: precision `0.7095`, recall `1.0000`, f1 `0.8300`, support `105`
- `springboard_rebound_only`: precision `0.7609`, recall `0.9929`, f1 `0.8615`, support `141`
- `platform_dive`: precision `0.0000`, recall `0.0000`, f1 `0.0000`, support `43`
- `noise_or_other`: precision `0.8333`, recall `0.1020`, f1 `0.1818`, support `49`

## Strongest Confusions

- `springboard_dive_vs_springboard_rebound_only`
  - held out: `evaluation_champigny_20260406-labelling`
  - accuracy: `0.9821`
  - confusion matrix: `[[42, 2], [0, 68]]`
- `platform_dive_vs_noise_or_other`
  - held out: `evaluation_insep_quick_9015_20260409_ui`
  - accuracy: `0.3944`
  - confusion matrix: `[[0, 43], [0, 28]]`
- `four_class`
  - macro F1: `0.4683`
  - confusion matrix: `[[105, 0, 0, 0], [0, 140, 0, 1], [43, 0, 0, 0], [0, 44, 0, 5]]`

## Decision

- `STOP_AND_ADDRESS_LABEL_GAPS`
