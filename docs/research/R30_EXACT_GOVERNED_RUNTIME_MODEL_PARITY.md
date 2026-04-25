# R30 Exact Governed Runtime Model Parity

R30 persisted the governed r9 compact nuisance weighted XGBoost model and wired runtime scoring to prefer this exact artifact over the prior bootstrapped proxy.

- Model artifact: `/Users/mcauchy/divesensei-app/.divesensei-runtime/models/r9_compact_nuisance_weighted/xgboost_model.json`
- Contract artifact: `/Users/mcauchy/divesensei-app/.divesensei-runtime/models/r9_compact_nuisance_weighted/contract.json`
- Classification: `exact_governed_runtime_parity_achieved`

The prior candidate-window risk was repaired in the follow-up parity check. Live runtime scoring now uses the governed platform/noise event window contract: proposal timestamp with `0.75s` pre-context and `2.25s` post-context. The r30 recheck showed exact runtime/offline agreement over the matched rows.
