# R29 Runtime/Offline Equivalence Audit

## Metrics

| metric | value |
| --- | --- |
| matched rows | 69 |
| pearson | 0.2461 |
| spearman | 0.2627 |
| threshold agreement | 0.8696 |
| runtime approved | 4 |
| offline approved | 7 |
| approve overlap | 1 |
| runtime dangerous | 1 |
| offline dangerous | 1 |

## Threshold Disagreements

```json
[
  {
    "row_key": "evaluation_insep_quick_9015_20260409_ui::det-0037",
    "runtime_session_id": "evaluation_r27_scorepath_insep_quick_v2",
    "runtime_detection_id": "det-0039",
    "label": "noise_or_other",
    "runtime_governed_r9_score": 0.9421256462285259,
    "offline_governed_r9_score": 0.239240825176239,
    "runtime_approved_v1": true,
    "offline_approved_v1": false,
    "runtime_visual_late_fusion_logreg_c0.5": 0.9991066586590726,
    "score_delta_runtime_minus_offline": 0.7028848210522869
  },
  {
    "row_key": "evaluation_insep_quick_9015_20260409_ui::det-0046",
    "runtime_session_id": "evaluation_r27_scorepath_insep_quick_v2",
    "runtime_detection_id": "det-0049",
    "label": "platform_dive",
    "runtime_governed_r9_score": 0.5735756905277525,
    "offline_governed_r9_score": 0.9384654760360718,
    "runtime_approved_v1": false,
    "offline_approved_v1": true,
    "runtime_visual_late_fusion_logreg_c0.5": 0.9804838094363286,
    "score_delta_runtime_minus_offline": -0.3648897855083193
  },
  {
    "row_key": "evaluation_insep_quick_9015_20260409_ui::det-0047",
    "runtime_session_id": "evaluation_r27_scorepath_insep_quick_v2",
    "runtime_detection_id": "det-0050",
    "label": "platform_dive",
    "runtime_governed_r9_score": 0.5739024354033255,
    "offline_governed_r9_score": 0.9411787986755371,
    "runtime_approved_v1": false,
    "offline_approved_v1": true,
    "runtime_visual_late_fusion_logreg_c0.5": 0.966721356144473,
    "score_delta_runtime_minus_offline": -0.3672763632722116
  },
  {
    "row_key": "evaluation_insep_quick_9015_20260409_ui::det-0050",
    "runtime_session_id": "evaluation_r27_scorepath_insep_quick_v2",
    "runtime_detection_id": "det-0053",
    "label": "platform_dive",
    "runtime_governed_r9_score": 0.6256532135311745,
    "offline_governed_r9_score": 0.9378089308738708,
    "runtime_approved_v1": false,
    "offline_approved_v1": true,
    "runtime_visual_late_fusion_logreg_c0.5": 0.9996838120512812,
    "score_delta_runtime_minus_offline": -0.3121557173426963
  },
  {
    "row_key": "evaluation_insep_quick_9015_20260409_ui::det-0057",
    "runtime_session_id": "evaluation_r27_scorepath_insep_quick_v2",
    "runtime_detection_id": "det-0061",
    "label": "platform_dive",
    "runtime_governed_r9_score": 0.8728051321971465,
    "offline_governed_r9_score": 0.9224345684051514,
    "runtime_approved_v1": false,
    "offline_approved_v1": true,
    "runtime_visual_late_fusion_logreg_c0.5": 0.9968152603788579,
    "score_delta_runtime_minus_offline": -0.04962943620800486
  },
  {
    "row_key": "evaluation_insep_quick_9015_20260409_ui::det-0069",
    "runtime_session_id": "evaluation_r27_scorepath_insep_quick_v2",
    "runtime_detection_id": "det-0072",
    "label": "platform_dive",
    "runtime_governed_r9_score": 0.6177239237304818,
    "offline_governed_r9_score": 0.950436532497406,
    "runtime_approved_v1": false,
    "offline_approved_v1": true,
    "runtime_visual_late_fusion_logreg_c0.5": 0.9512694258394299,
    "score_delta_runtime_minus_offline": -0.3327126087669242
  },
  {
    "row_key": "evaluation_champigny_20260406-labelling::det-0004",
    "runtime_session_id": "evaluation_r27_scorepath_champigny_proxy",
    "runtime_detection_id": "det-0004",
    "label": "platform_dive",
    "runtime_governed_r9_score": 0.9797077610522851,
    "offline_governed_r9_score": 0.35440459847450256,
    "runtime_approved_v1": true,
    "offline_approved_v1": false,
    "runtime_visual_late_fusion_logreg_c0.5": 0.9989534880949669,
    "score_delta_runtime_minus_offline": 0.6253031625777825
  },
  {
    "row_key": "evaluation_champigny_20260406-labelling::det-0007",
    "runtime_session_id": "evaluation_r27_scorepath_champigny_proxy",
    "runtime_detection_id": "det-0007",
    "label": "noise_or_other",
    "runtime_governed_r9_score": 0.323963501406053,
    "offline_governed_r9_score": 0.9423382878303528,
    "runtime_approved_v1": false,
    "offline_approved_v1": true,
    "runtime_visual_late_fusion_logreg_c0.5": 0.27347653264772226,
    "score_delta_runtime_minus_offline": -0.6183747864242998
  },
  {
    "row_key": "evaluation_champigny_20260406-labelling::det-0110",
    "runtime_session_id": "evaluation_r27_scorepath_champigny_proxy",
    "runtime_detection_id": "det-0112",
    "label": "platform_dive",
    "runtime_governed_r9_score": 0.9989010333987904,
    "offline_governed_r9_score": 0.44732511043548584,
    "runtime_approved_v1": true,
    "offline_approved_v1": false,
    "runtime_visual_late_fusion_logreg_c0.5": 0.9975098564507296,
    "score_delta_runtime_minus_offline": 0.5515759229633046
  }
]
```

## Root Cause

- Runtime uses a bootstrapped logistic proxy model.
- Offline governed reference uses the source-weighted r9 XGBoost compact nuisance model.
- Feature vectors, scaling, model family, and training source are not identical.
