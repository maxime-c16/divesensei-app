# Springboard Failure Analysis

## Core Finding

- true springboard_dive rows in reviewed baseline: `78`
- correctly classified springboard_dive rows: `0`
- springboard_dive -> springboard_rebound_only errors: `78`

## Session Concentration

- failures by session: `{"evaluation_champigny_20260406-labelling": 33, "evaluation_insep_15min_validated": 45}`
- all springboard_dive truth rows by session: `{"evaluation_champigny_20260406-labelling": 33, "evaluation_insep_15min_validated": 45}`

## Error Pattern

- legacy label / subtype on failures: `{"dive|None": 78}`
- suggestion reason on failures: `{"insufficient_context_uncertain": 20, "rebound_context_plus_delayed_entry": 45, "springboard_dive_without_rebound_context": 13}`
- support flags on failures: `{"has_entry": {"true": 78}, "has_rebound": {"false": 33, "true": 45}, "no_rebound": {"false": 45, "true": 33}}`
- anchor/window summary: `{"mean_anchor_minus_proposal": 0.0, "mean_window_end": 1028.7630256410257, "mean_window_start": 1025.7630256410257, "window_lengths": {"3.0": 78}}`

## Direct Diagnostic

- The model does not recover any `springboard_dive` in the reviewed four-class fit. All springboard_dive rows are sent to `springboard_rebound_only` or, in a smaller set, to `noise_or_other`/`platform_dive` depending on the fold context.
- The failure rows are concentrated in springboard-heavy sessions, not in the platform session.
- Their support metadata splits into two dominant patterns:
  - `rebound_context_plus_delayed_entry`
  - `springboard_dive_without_rebound_context`
- That means the current 3.0s proposal-centered window is not separating “true dive” from “rebound-only” structure in a way the linear baseline can use.

## Comparison Against Correctly Classified Springboard Dives

- There are no correctly classified `springboard_dive` rows in the current reviewed baseline fold run, so the comparison set is empty.
- This is itself diagnostic: the current feature/window setup gives the baseline no learnable boundary for the springboard dive class.

## Ranked Bottlenecks

1. `C. springboard window length/content is wrong`
2. `B. springboard anchor strategy is wrong`
3. `E. current baseline model is too weak`
4. `D. mixed-session rows should be held out from first baseline`
5. `A. more springboard review labels needed`

## Best Next Action

- The single best next action is to redesign the springboard event window so it captures a more discriminative entry-vs-rebound context, then re-test the same simple baseline before adding more labels or model complexity.

## Top Failure Rows

- `evaluation_champigny_20260406-labelling` `det-0008` final `springboard_dive` predicted `springboard_rebound_only` reason `springboard_dive_without_rebound_context` has_rebound=False has_entry=True no_rebound=True window=631.874–634.874
- `evaluation_champigny_20260406-labelling` `det-0009` final `springboard_dive` predicted `springboard_rebound_only` reason `springboard_dive_without_rebound_context` has_rebound=False has_entry=True no_rebound=True window=714.706–717.706
- `evaluation_champigny_20260406-labelling` `det-0016` final `springboard_dive` predicted `springboard_rebound_only` reason `rebound_context_plus_delayed_entry` has_rebound=True has_entry=True no_rebound=False window=825.426–828.426
- `evaluation_champigny_20260406-labelling` `det-0020` final `springboard_dive` predicted `springboard_rebound_only` reason `rebound_context_plus_delayed_entry` has_rebound=True has_entry=True no_rebound=False window=896.594–899.594
- `evaluation_champigny_20260406-labelling` `det-0023` final `springboard_dive` predicted `springboard_rebound_only` reason `rebound_context_plus_delayed_entry` has_rebound=True has_entry=True no_rebound=False window=959.618–962.618
- `evaluation_champigny_20260406-labelling` `det-0028` final `springboard_dive` predicted `springboard_rebound_only` reason `rebound_context_plus_delayed_entry` has_rebound=True has_entry=True no_rebound=False window=1014.594–1017.594
- `evaluation_champigny_20260406-labelling` `det-0035` final `springboard_dive` predicted `springboard_rebound_only` reason `rebound_context_plus_delayed_entry` has_rebound=True has_entry=True no_rebound=False window=1104.066–1107.066
- `evaluation_champigny_20260406-labelling` `det-0040` final `springboard_dive` predicted `springboard_rebound_only` reason `rebound_context_plus_delayed_entry` has_rebound=True has_entry=True no_rebound=False window=1176.898–1179.898
- `evaluation_champigny_20260406-labelling` `det-0043` final `springboard_dive` predicted `springboard_rebound_only` reason `rebound_context_plus_delayed_entry` has_rebound=True has_entry=True no_rebound=False window=1241.314–1244.314
- `evaluation_champigny_20260406-labelling` `det-0047` final `springboard_dive` predicted `springboard_rebound_only` reason `rebound_context_plus_delayed_entry` has_rebound=True has_entry=True no_rebound=False window=1318.626–1321.626
- `evaluation_champigny_20260406-labelling` `det-0048` final `springboard_dive` predicted `springboard_rebound_only` reason `springboard_dive_without_rebound_context` has_rebound=False has_entry=True no_rebound=True window=1571.650–1574.650
- `evaluation_champigny_20260406-labelling` `det-0051` final `springboard_dive` predicted `springboard_rebound_only` reason `rebound_context_plus_delayed_entry` has_rebound=True has_entry=True no_rebound=False window=1642.194–1645.194
- `evaluation_champigny_20260406-labelling` `det-0056` final `springboard_dive` predicted `springboard_rebound_only` reason `rebound_context_plus_delayed_entry` has_rebound=True has_entry=True no_rebound=False window=1715.554–1718.554
- `evaluation_champigny_20260406-labelling` `det-0062` final `springboard_dive` predicted `springboard_rebound_only` reason `springboard_dive_without_rebound_context` has_rebound=False has_entry=True no_rebound=True window=1809.746–1812.746
- `evaluation_champigny_20260406-labelling` `det-0068` final `springboard_dive` predicted `springboard_rebound_only` reason `rebound_context_plus_delayed_entry` has_rebound=True has_entry=True no_rebound=False window=1876.706–1879.706
- `evaluation_champigny_20260406-labelling` `det-0070` final `springboard_dive` predicted `springboard_rebound_only` reason `rebound_context_plus_delayed_entry` has_rebound=True has_entry=True no_rebound=False window=1973.586–1976.586
- `evaluation_champigny_20260406-labelling` `det-0074` final `springboard_dive` predicted `springboard_rebound_only` reason `rebound_context_plus_delayed_entry` has_rebound=True has_entry=True no_rebound=False window=2064.562–2067.562
- `evaluation_champigny_20260406-labelling` `det-0078` final `springboard_dive` predicted `springboard_rebound_only` reason `rebound_context_plus_delayed_entry` has_rebound=True has_entry=True no_rebound=False window=2142.594–2145.594
- `evaluation_champigny_20260406-labelling` `det-0084` final `springboard_dive` predicted `springboard_rebound_only` reason `rebound_context_plus_delayed_entry` has_rebound=True has_entry=True no_rebound=False window=2227.330–2230.330
- `evaluation_champigny_20260406-labelling` `det-0085` final `springboard_dive` predicted `springboard_rebound_only` reason `springboard_dive_without_rebound_context` has_rebound=False has_entry=True no_rebound=True window=2304.546–2307.546
