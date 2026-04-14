# Platform/Noise ES4 Model Benchmark

| model | AUC | macro F1 | accuracy | platform recall | noise recall | confusion |
|---|---:|---:|---:|---:|---:|---|
| numpy_logistic_reference | 0.6100 | 0.5489 | 0.5500 | 0.6000 | 0.5000 | `[[6, 4], [5, 5]]` |
| sklearn_logistic_l2 | 0.6100 | 0.5489 | 0.5500 | 0.6000 | 0.5000 | `[[6, 4], [5, 5]]` |
| xgboost_gbdt | 0.7100 | 0.6970 | 0.7000 | 0.8000 | 0.6000 | `[[8, 2], [4, 6]]` |

- best candidate: `xgboost_gbdt`