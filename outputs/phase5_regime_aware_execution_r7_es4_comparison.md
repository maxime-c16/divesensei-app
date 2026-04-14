# Phase 5 r7-es4 Comparison

- final decision: `PHASE5_R7_ES4_PASS`
- main reason: `all guardbands passed.`

## Platform/noise transfer check

- logistic regime r4 AUC/macro F1: `0.5100 / 0.3732`
- ES4 benchmark xgboost AUC/macro F1: `0.7100 / 0.6970`
- r7-es4 regime AUC/macro F1: `0.7100 / 0.6970`
- gains transferred cleanly: `True`
- hard rows improved vs r4 (noise->platform FP reduced): `True`

## Springboard regression check

- springboard AUC r4 -> r7: `0.7745 -> 0.7745`
- springboard macro F1 r4 -> r7: `0.5048 -> 0.5048`
- springboard FN r4 -> r7: `32 -> 32`
- springboard FP r4 -> r7: `0 -> 0`
