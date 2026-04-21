# R30 Exact Governed Runtime Model Parity

R30 persisted the governed r9 compact nuisance weighted XGBoost model and wired runtime scoring to prefer this exact artifact over the prior bootstrapped proxy.

- Model artifact: `/Users/mcauchy/divesensei-app/.divesensei-runtime/models/r9_compact_nuisance_weighted/xgboost_model.json`
- Contract artifact: `/Users/mcauchy/divesensei-app/.divesensei-runtime/models/r9_compact_nuisance_weighted/contract.json`
- Classification: `exact_governed_runtime_parity_achieved`

The remaining risk is window parity: live runtime scoring currently uses candidate windows, while the governed offline training contract used event-window manifest windows.
