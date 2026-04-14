# Phase 5 Regime-Aware Execution (r7-es4)

- decision: `PHASE5_R7_ES4_PASS`
- main reason: `all guardbands passed.`

## Input integrity

- frozen row lists unchanged: `True`
- train/holdout overlap row_keys: `0`
- springboard unchanged from r4: `True`
- platform/noise representation unchanged from ES4 input: `True`
- only platform/noise model family changed: `True` (`xgboost_gbdt`)

## Guardband checks

| track | check | threshold | value | status |
|---|---|---|---|---|
| `springboard_track` | `success_mean_auc_min` | `0.52` | `0.7745098039215687` | **PASS** |
| `springboard_track` | `success_mean_macro_f1_min` | `0.5` | `0.5047619047619047` | **PASS** |
| `springboard_track` | `success_champigny_macro_f1_min` | `0.44` | `0.5047619047619047` | **PASS** |
| `platform_noise_track` | `success_mean_auc_min` | `0.66` | `0.71` | **PASS** |
| `platform_noise_track` | `success_mean_macro_f1_min` | `0.5` | `0.696969696969697` | **PASS** |
| `platform_noise_track` | `success_champigny_macro_f1_min` | `0.64` | `0.696969696969697` | **PASS** |
| `catastrophic` | `springboard_all_dive_predicted_as_rebound` | `must_not_trigger` | `At least one holdout springboard_dive predicted correctly.` | **PASS** |
| `catastrophic` | `platform_holdout_recall_below_0p75` | `must_not_trigger` | `Holdout platform_dive recall=0.8000` | **PASS** |
