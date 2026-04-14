# Handoff — 2026-04-14 ES4 model-family benchmark

This handoff captures the current session state for continuing Phase 5 regime-aware work on the school machine.

## 1) What was done this session

1. Added next-cycle meta-experiment specification:
   - `docs/research/DETECTOR_MENTAL_MODEL_META_EXPERIMENTS.md`
2. Executed the first concrete ES4 benchmark (platform/noise, frozen-policy):
   - `benchmarks/platform_noise_es4_dataset.py`
   - `benchmarks/platform_noise_es4_model_benchmark.py`
3. Materialized local outputs (ignored by git, but present under `outputs/`):
   - `outputs/platform_noise_es4_dataset_summary.json`
   - `outputs/platform_noise_es4_dataset_summary.md`
   - `outputs/platform_noise_es4_model_benchmark.json`
   - `outputs/platform_noise_es4_model_benchmark.md`
   - helper artifacts: `platform_noise_es4_dataset.npz`, `platform_noise_es4_dataset_rows.json`

## 2) Frozen constraints respected

- Detector behavior unchanged (legacy validated detector still proposal-generator only).
- Taxonomy unchanged.
- Labels unchanged.
- Springboard configuration unchanged.
- Platform/noise split policy taken from frozen `outputs/phase5_regime_manifest_lists.json`.
- Holdout leakage prevented (`train_holdout_overlap_row_keys = 0`).

## 3) ES4 dataset snapshot

- Train rows: 53
- Scored validation rows: 20
- Reporting-only rows excluded: 10
- Train label counts: `{"noise_or_other": 20, "platform_dive": 33}`
- Scored validation label counts: `{"noise_or_other": 10, "platform_dive": 10}`

Feature schema used:
- baseline scalar features
- accepted platform/noise feature family (r1 + r2 + r4) included as-is
- total columns: 18

## 4) ES4 benchmark result summary

Compared on frozen scored validation slice:

1. `numpy_logistic_reference`
2. `sklearn_logistic_l2`
3. `xgboost_gbdt`

Best candidate: `xgboost_gbdt`

- AUC: 0.75
- macro F1: 0.6970
- accuracy: 0.70
- platform recall: 0.80
- noise recall: 0.60
- confusion: `[[8, 2], [4, 6]]`
- noise->platform FP: 4
- platform->noise FN: 2

Decision emitted in benchmark artifact:
- `ES4_PASS_PROMOTE_TO_PHASE5_RERUN`

## 5) Repo docs to treat as current reference

- `docs/research/PHASE5_FINAL_STATUS.md`
- `docs/HANDOFF_2026-04-13_NEAR_PASS_CLOSURE.md`
- `docs/research/DETECTOR_MENTAL_MODEL_META_EXPERIMENTS.md`
- `docs/HANDOFF_2026-04-14_ES4_MODEL_FAMILY_BENCHMARK.md` (this file)

## 6) Repro command sequence (local)

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install scikit-learn xgboost
brew install libomp
python benchmarks/platform_noise_es4_dataset.py
python benchmarks/platform_noise_es4_model_benchmark.py
make compile smoke-help
```

## 7) Next step at school

Run the same two ES4 scripts on the school machine and compare outputs against local values above.  
If consistent, proceed to the next governed step defined in the meta-experiment doc (ES5 ranking-aware benchmark), still under frozen detector/taxonomy/label/springboard constraints.

