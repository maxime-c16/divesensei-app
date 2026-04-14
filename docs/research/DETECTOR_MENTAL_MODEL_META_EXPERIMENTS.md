# Detector Mental-Model Meta-Experiments (Next Cycle)

Date: 2026-04-14

## Executive summary

Phase 5 ended as a **near-pass**: springboard is a real pass and stable, while platform/noise gained strong ranking signal (`AUC=0.81`) but still fails joint operating guardrails under current constraints. Repo evidence shows this is no longer a “one more feature” problem; it is now a **pipeline mental-model and decision-policy problem**: representation improved, but boundary/ordering/policy interactions still block pass.

The next cycle should run a **disciplined meta-experiment program** with explicit path families, pass/kill rules, and governance-grade evidence. The best first path is: **stronger platform/noise model family on frozen data/labels with strict uncertainty-aware evaluation**, followed by **ranking-aware objectives**, then **calibration/constrained policy**.

---

## Current verified state (repo-grounded)

### Verified from local artifacts

1. **Detector is frozen baseline and proposal-generator only** (`docs/DETECTOR_STATUS_2026-04.md`).
2. **Taxonomy is fixed** to 4 classes in event-window line (`springboard_dive`, `springboard_rebound_only`, `platform_dive`, `noise_or_other`) and represented in event manifest/review workflows (`src/divesensei/workflows/event_window_manifest.py`, `event_review_support.py`).
3. **Anchor-policy lesson is confirmed**: springboard needed anchor correction (`earliest_strong_peak_in_local_cluster`) while platform rows remained proposal-centered (`outputs/springboard_window_anchor_experiment.md`, `src/divesensei/workflows/event_review_support.py`).
4. **Phase 5 regime-aware runs**:
   - r4 fail: platform/noise weak (`AUC=0.51`, `macro F1=0.3732`)
   - r5 fail: platform/noise improved (`AUC=0.64`)
   - r6 fail near-pass: platform/noise `AUC=0.65`, `macro F1=0.5833`, springboard stable pass
   (`outputs/phase5_regime_aware_execution_r{4,5,6}.md`, `..._r6_comparison.md`).
5. **Platform/noise feature progression**:
   - baseline `AUC=0.51` -> r1 `0.64` -> r2 `0.65` -> r3 regression `0.64` -> r4 representation jump `0.81`
   (`outputs/platform_noise_feature_probe*.md`).
6. **Operating-point analysis confirms no feasible threshold** that satisfies all required guardrails simultaneously (platform recall floor + macro F1 floor + strict FP improvement vs r4 default) (`outputs/platform_noise_r4_operating_point_analysis.md`).
7. **Cycle closure is near-pass, not pass** (`docs/research/PHASE5_FINAL_STATUS.md`, `docs/HANDOFF_2026-04-13_NEAR_PASS_CLOSURE.md`, `outputs/phase5_near_pass_summary.json`).

### Missing on this machine (not silently assumed)

Not present locally:  
`docs/HANDOFF_2026-04-09_REGION_TIEBREAK.md`, `docs/HANDOFF_2026-04-10_VALIDATED_BRANCH_AND_SPRINGBOARD_PIVOT.md`, `docs/HANDOFF_2026-04-13_PHASE5_R4_PLATFORM_NOISE_NEXT.md`, `docs/research/PHASE5_REGIME_AWARE_PROTOCOL.md`, `docs/research/PHASE5_REGIME_AWARE_FREEZE.md`, `docs/research/PHASE5_STATUS_RESOLUTION.md`, `outputs/phase5_guardbands.*`, `outputs/event_classifier_baseline_reviewed_anchorpropfix.*`, `outputs/full_four_class_anchorfix_gap_analysis.*`, and several older springboard probe artifacts.

This plan therefore treats those missing artifacts as unverified and uses available Phase 5 outputs as the canonical evidence base.

---

## Part A — Updated detector+classifier mental model

## A1) Operational model of the current pipeline

1. **Detector layer (frozen):** proposal generation from validated legacy detector branch.  
2. **Event-review/event-window layer:** reviewed manifests + regime-aware anchor policy generate event windows.  
3. **Track-specific classification layer:** springboard and platform/noise evaluated via frozen row lists (`outputs/phase5_regime_manifest_lists.json`).  
4. **Decision layer:** fixed guardbands + catastrophic checks + threshold policy.

## A2) What is frozen vs flexible

| Area | Frozen now | Flexible next cycle |
|---|---|---|
| Detector behavior/threshold logic | Yes | No (this cycle) |
| Taxonomy/review labels | Yes | No (except future data growth, not relabel rewrite) |
| Springboard representation | Yes (`probe_r1_only`) | No in early next-cycle steps |
| Platform/noise train/holdout slice policy | Yes for comparability | Yes initially; optional expansion only as explicit later family |
| Classifier family | Historically fixed logistic | **Primary next-cycle candidate for change** |
| Operating-point policy | Currently thresholded | Flexible via constrained policy analysis |

## A3) What anchor-policy and springboard work taught us

1. Wrong anchor/window can fully suppress learnability even with same model family.
2. Springboard became learnable after anchor correction; window-length tweaks mattered less than anchor semantics.
3. Therefore, **representation semantics can dominate performance more than small model tweaks**.

## A4) What platform/noise work taught us

1. The initial blocker was real and measurable on a valid frozen scored slice.
2. Feature work moved ranking substantially (`AUC 0.51 -> 0.81`), so “no signal” is false.
3. Residual error cluster is mixed (tonal + diffuse handling + ambiguous), so narrow single-axis fixes fail (r3).
4. With strong representation, pass is now blocked by **boundary/ordering/policy constraints**, not pure feature absence.

## A5) Bottleneck ranking (current best interpretation)

1. **Decision boundary/model family capacity (platform/noise)** — likely primary remaining technical bottleneck.
2. **Operating-point policy under constraints** — proven blocker in current family.
3. **Residual negative-cluster heterogeneity** — still contributes.
4. **Validation governance/uncertainty with tiny holdout** — affects confidence of close calls.
5. **Proposal generation** — no direct evidence it is the dominant current blocker for platform/noise near-pass.
6. **Label ambiguity** — lower-ranked in current frozen evidence.

## A6) Accepted conclusions vs open hypotheses

### Accepted (supported by repo evidence)

1. Detector remains frozen baseline proposal generator.
2. Springboard is passing/stable in regime-aware line.
3. Platform/noise has strong ranking improvement but guardrail failure remains.
4. Threshold tuning alone is insufficient under current constraints.
5. Further tiny one-off feature/threshold tweaks are low-value.

### Open hypotheses (to be tested next cycle)

1. A stronger tabular model family can convert ranking gains into guardrail-feasible classification.
2. Ranking-aware objectives can reduce residual score-order overlap better than plain logistic loss.
3. Calibrated constrained decision policy may recover acceptable operating behavior once rank separation is improved.
4. Frozen holdout uncertainty may make strict pass/fail thresholds overconfident without CI framing.
5. Pretrained embeddings may improve robustness on mixed negative cluster without detector changes.

---

## Part B — Path-family inventory

## B1) Family inventory table

| Family | Why plausible | Repo evidence motivator | External/community motivator | Main risks | Success signal | Kill signal |
|---|---|---|---|---|---|---|
| 1. Detector mental-model / proposal-family diagnostics | Could still hide upstream structure limits | Detector architecture coupling documented in status doc | SED practice emphasizes proposal quality/coverage diagnostics | Drifts into forbidden detector edits | Better causal attribution without detector changes | No new explanatory power after row-level diagnostics |
| 2. Window/anchor/temporal representation | Previously decisive for springboard | Anchor correction unlocked springboard | Audio-event literature supports timing/onset structure importance | Reopening solved springboard line | Platform/noise gain under fixed model from temporal representation only | Repeats r3-style narrow tweaks or harms recall |
| 3. Hand-crafted feature families | Already delivered large gains | r1/r2/r4 progression | MIR features (spectral contrast/onset/tempogram) are standard | Diminishing returns; local overfit | Consistent residual-cluster reduction with no recall collapse | One-off features that improve one metric but destabilize boundary |
| 4. Stronger tabular model family | Next likely bottleneck after representation | r4 strong AUC but failed guardrails under logistic | LightGBM/XGBoost robust on tabular interactions | Overfitting on tiny n | Better joint guardrail feasibility on frozen slice + stability | Gains vanish under resampling/CI |
| 5. Ranking-aware/AUC-first training | AUC blocker is ordering overlap | residual/operating-point analyses emphasize ranking | Pairwise ranking/LTR methods explicitly target ordering | Query-group mismatch, instability | Higher AUC + reduced hard-negative overlap | AUC gain with recall collapse or unstable behavior |
| 6. Pretrained embedding path | Mixed residual cluster may need richer semantics | Hand-crafted path reached near-pass ceiling | OpenL3/PANNs transfer for small labeled tasks | Dependency/compute/domain mismatch | Robust cluster separation + reduced feature brittleness | No gain over strong tabular baseline after fair protocol |
| 7. Calibration / constrained decision policy | Current blocker is policy tradeoff | threshold sweep showed infeasible set in current setup | Calibration + constrained optimization are standard | Misused as fake progress without rank gains | Feasible operating region under explicit constraints | Still infeasible after better ranking model |
| 8. Evaluation-governance / uncertainty | Tiny holdout makes boundary claims fragile | close-threshold near-pass history | DeLong/bootstrap CI common ROC practice | Process overhead, false certainty from misuse | Pass criteria become statistically defensible | CI remains too wide for any reliable promotion claim |
| 9. Semi-supervised / augmentation | Could address heterogeneity with limited labels | residual cluster still mixed | DCASE-style weak/heterogeneous training norms | Leakage risk, large complexity jump | Gains beyond supervised baselines with strict leakage control | No reliable gain after controlled pilot |
| 10. Detector-side alternative interpretation (allowed-only) | Keep detector frozen but reinterpret proposal evidence downstream | status docs caution against unsafe detector injections | Common pattern: “observe more, edit less” before architecture changes | Hidden detector edits by accident | Better diagnostics and policy insights, zero detector changes | No incremental evidence over existing diagnostics |

---

## Part C — Experiment sets (benchmark program)

Global rule for all sets unless explicitly overridden:

- **Frozen constraints:** detector behavior, taxonomy, reviewed labels, springboard accepted configuration, and baseline slice definitions for comparability.
- **Forbidden:** silent detector edits, taxonomy relabeling, retrospective metric reinterpretation, tuning on scored holdout.
- **Decision classes:** `PASS`, `NEAR_PASS`, `FAIL/KILL`.

## ES1 — Proposal-family observability and failure-surface map

- **Question:** Are residual failures dominated by missing/poor proposals or by downstream representation/boundary?
- **Hypothesis:** Current residual platform/noise near-pass is mostly downstream, not proposal absence.
- **Required inputs:** frozen reviewed manifests + `phase5_regime_manifest_lists.json` + phase5 execution/probe outputs.
- **Data split rules:** no retraining required; row-level diagnostics over frozen train/holdout.
- **Allowed changes:** diagnostics scripts only.
- **Forbidden changes:** any proposal generation or threshold behavior changes.
- **Expected artifacts:**
  - `outputs/metaexp_es1_proposal_failure_surface.json/.md`
  - `outputs/metaexp_es1_row_level_tracebacks.jsonl`
- **Metrics:** proposal coverage at event-level truth rows, downstream misranking counts conditional on proposal presence.
- **Decision rule:**  
  - **PASS:** evidence clearly attributes >70% residual errors to downstream boundary/ranking with proposal present.  
  - **KILL detector-rework path for now:** if proposal-miss contribution remains minor.
- **Interpretation:** establishes whether detector-side work is warranted this cycle.
- **Unlock if pass:** prioritize ES4/ES5 over detector rework.
- **Fallback if fail:** schedule bounded detector-side diagnostic family before model upgrades.

## ES2 — Window/anchor temporal-ablation sanity set (platform/noise only)

- **Question:** Is platform/noise still materially limited by temporal framing under frozen anchor policy?
- **Hypothesis:** Minor temporal variants will not beat model-family upgrades now.
- **Required inputs:** same frozen rows, same labels, same feature extractor backbone.
- **Data split rules:** frozen train/holdout; no row movement.
- **Allowed changes:** bounded temporal representation variants in offline extraction only.
- **Forbidden:** springboard anchor modifications; full taxonomy retraining.
- **Expected artifacts:** `outputs/metaexp_es2_temporal_ablation.json/.md`.
- **Metrics:** AUC, macro F1, platform recall, noise FP, confusion.
- **Decision rule:**  
  - **PASS temporal path:** if a temporal-only variant beats current accepted representation and improves guardrail feasibility.  
  - **KILL:** if gains are marginal/unstable vs r4 baseline.
- **Unlock if pass:** targeted representation refinement before model-family shift.
- **Fallback if fail:** proceed to ES4.

## ES3 — Hand-crafted feature bundle reproducibility and robustness check

- **Question:** Are accepted r4 gains robust under repeated resampling, not just single holdout outcome?
- **Hypothesis:** r4 is a real representation gain, but still insufficient alone for guardrail pass.
- **Required inputs:** frozen manifests + audio windows + r2/r4 probe feature pipelines.
- **Data split rules:** frozen official holdout + repeated stratified resampling on train only.
- **Allowed changes:** robustness reporting only.
- **Forbidden:** adding new ad hoc feature sweeps.
- **Expected artifacts:** `outputs/metaexp_es3_feature_robustness.json/.md`.
- **Metrics:** distribution of AUC/macro F1 deltas (r4-r2), variance/CI.
- **Decision rule:**  
  - **PASS:** r4 > r2 consistently in ranking metrics with acceptable variance.  
  - **NEAR_PASS:** directional but unstable.  
  - **KILL feature-iteration:** if unstable/noisy and not decision-improving.
- **Unlock if pass:** use r4 representation as fixed base for model-family tests.
- **Fallback if fail:** freeze at r2 and move immediately to ES4.

## ES4 — Stronger tabular model-family benchmark (primary)

- **Question:** Can non-linear tabular models convert current representation signal into guardrail-feasible behavior?
- **Hypothesis:** Gradient-boosted trees on frozen feature sets outperform logistic on residual mixed cluster.
- **Required inputs:** immutable dataset artifact containing row keys, labels, features from accepted representation.
- **Data split rules:** frozen official holdout untouched; training uses nested CV for model selection.
- **Allowed changes:** model family and hyperparameters only.
- **Forbidden:** detector/taxonomy/label/split changes.
- **Expected artifacts:**
  - `outputs/metaexp_es4_model_family_results.json/.md`
  - `outputs/metaexp_es4_feature_interaction_report.md`
  - `outputs/metaexp_es4_holdout_predictions.jsonl`
- **Metrics:** AUC, macro F1, platform recall, noise recall, FP/FN counts, calibration diagnostics.
- **Decision rule:**  
  - **PASS:** beats logistic baseline on AUC and produces at least one feasible operating region candidate under guardrails (validated in ES7).  
  - **NEAR_PASS:** AUC gain without guardrail-feasible operating point.  
  - **KILL:** no robust gain over logistic+r4.
- **Unlock if pass:** ES7 constrained policy finalization and Phase 5 r7 candidate integration.
- **Fallback if fail:** ES5 (ranking-aware) or ES6 (embeddings).

## ES5 — Ranking-aware / AUC-first objective benchmark

- **Question:** Does pairwise/ranking optimization reduce ordering overlap better than standard classification loss?
- **Hypothesis:** Ranking-aware training improves hard-negative ordering and AUC stability.
- **Required inputs:** same immutable feature dataset as ES4; optional grouping by session root.
- **Data split rules:** same frozen holdout; strict no-leak query grouping.
- **Allowed changes:** objective function (`rank:pairwise` / equivalent), scoring policy.
- **Forbidden:** changing labels or manually weighting holdout errors.
- **Expected artifacts:** `outputs/metaexp_es5_ranking_objective_results.json/.md`.
- **Metrics:** AUC, pairwise ordering error on residual cluster, guardrail metrics at candidate thresholds.
- **Decision rule:**  
  - **PASS:** statistically supported AUC gain and reduced residual overlap vs ES4 best classifier.  
  - **KILL:** no stable ordering benefit or severe recall harm.
- **Unlock if pass:** ES7 policy calibration on ranking model.
- **Fallback if fail:** prioritize ES6 or governance stop decision.

## ES6 — Pretrained embedding benchmark with shallow heads

- **Question:** Do pretrained audio embeddings provide more robust separation for mixed noise cluster than hand-crafted features?
- **Hypothesis:** frozen embeddings + shallow head improve generalization on small reviewed data.
- **Required inputs:** frozen event windows, same row keys, embedder configs (e.g., OpenL3/PANNs).
- **Data split rules:** same frozen holdout; embeddings cached per row key.
- **Allowed changes:** representation source and head model (logistic/GBDT).
- **Forbidden:** end-to-end detector retraining; holdout-tuned embedding selection.
- **Expected artifacts:**
  - `outputs/metaexp_es6_embeddings_catalog.json`
  - `outputs/metaexp_es6_embedding_head_results.json/.md`
  - `outputs/metaexp_es6_residual_cluster_deltas.md`
- **Metrics:** AUC/macro F1/guardrails + residual-cluster error deltas.
- **Decision rule:**  
  - **PASS:** clear robust gain over ES4/ES5 best on both ranking and practical guardrails.  
  - **KILL:** no gain after controlled comparison.
- **Unlock if pass:** promote embedding path to integration candidate.
- **Fallback if fail:** proceed to ES8 governance decision.

## ES7 — Calibration + constrained operating-policy search

- **Question:** Given best model from ES4/ES5/ES6, is there a defensible operating point satisfying project constraints?
- **Hypothesis:** with stronger ranking model, constrained threshold/calibration can satisfy recall/FP guardrails.
- **Required inputs:** best model scores on calibration split + frozen holdout predictions.
- **Data split rules:** calibration performed off holdout (CV/calibration split); holdout for final evaluation only.
- **Allowed changes:** post-hoc calibration and constrained threshold search.
- **Forbidden:** re-optimizing feature/model on holdout outcomes.
- **Expected artifacts:** `outputs/metaexp_es7_policy_frontier.json/.md`.
- **Metrics:** feasibility map over thresholds, Brier/reliability, guardrail satisfaction.
- **Decision rule:**  
  - **PASS:** at least one operating point satisfies all guardrails with stable behavior.  
  - **NEAR_PASS:** close frontier but no feasible point.  
  - **KILL:** policy path cannot rescue even improved rankers.
- **Unlock if pass:** scientifically defensible Phase 5 r7 integration.
- **Fallback if fail:** ES8 governance decision for near-pass closure.

## ES8 — Evaluation governance and uncertainty quantification

- **Question:** Are near-threshold decisions statistically defensible on current holdout size?
- **Hypothesis:** CI-aware governance will reduce false “pass/fail by noise” outcomes.
- **Required inputs:** all candidate-model holdout scores and predictions.
- **Data split rules:** no training; post-hoc analysis only.
- **Allowed changes:** reporting/protocol criteria updates.
- **Forbidden:** retroactive metric cherry-picking.
- **Expected artifacts:**
  - `outputs/metaexp_es8_auc_uncertainty.json/.md`
  - `outputs/metaexp_es8_guardband_governance_proposal.md`
- **Metrics:** DeLong/Bootstrap AUC intervals, sensitivity to row perturbations.
- **Decision rule:**  
  - **PASS:** governance rubric accepted and reproducible.  
  - **KILL:** if uncertainty too high to support promotion claims.
- **Unlock if pass:** stable promotion/near-pass policy for next cycles.
- **Fallback if fail:** explicit “insufficient evidence” status + data expansion requirement.

## ES9 — Optional semi-supervised / augmentation pilot

- **Question:** Can unlabeled proposal pools improve robustness after supervised paths plateau?
- **Hypothesis:** modest SSL/augmentation can reduce mixed-cluster variance.
- **Required inputs:** labeled frozen set + unlabeled proposal windows with provenance.
- **Data split rules:** strict leakage prevention by session and row lineage.
- **Allowed changes:** SSL/augmentation modules only in pilot.
- **Forbidden:** using unlabeled data that leaks holdout semantics.
- **Expected artifacts:** `outputs/metaexp_es9_ssl_pilot.json/.md`.
- **Metrics:** delta vs ES4/ES5/ES6 best, variance under resampling.
- **Decision rule:**  
  - **PASS:** meaningful gain with controlled leakage risk.  
  - **KILL:** complexity high, gain low.
- **Unlock if pass:** larger SSL program.
- **Fallback if fail:** stop SSL line for current cycle.

## ES10 — Detector-side alternative interpretation (no behavior change)

- **Question:** Can richer interpretation of frozen detector outputs improve downstream decisions without detector edits?
- **Hypothesis:** additional proposal diagnostics/provenance features may help policy attribution, not detector changes.
- **Required inputs:** candidate diagnostics, proposal metadata, event windows.
- **Data split rules:** unchanged from frozen policy.
- **Allowed changes:** derived metadata features from existing artifacts only.
- **Forbidden:** changing proposal-generation logic, suppression, tie-breaks.
- **Expected artifacts:** `outputs/metaexp_es10_detector_interpretation.json/.md`.
- **Metrics:** attribution clarity and incremental predictive utility.
- **Decision rule:**  
  - **PASS:** improved explanatory/decision power with zero detector edits.  
  - **KILL:** no measurable utility.
- **Unlock if pass:** add interpretation channels to ES4+ datasets.
- **Fallback if fail:** drop detector-interpretation line this cycle.

---

## Part D — Priority ordering (execution roadmap)

## Recommended execution order

1. **ES4 — Stronger tabular model-family benchmark (top priority).**  
2. **ES5 — Ranking-aware objective benchmark.**  
3. **ES7 — Calibration + constrained operating-policy search** (on ES4/ES5 winner).  
4. **ES8 — Governance/uncertainty formalization** (in parallel with ES7 reporting).  
5. **ES6 — Pretrained embeddings** (if ES4/ES5 still near-pass).  
6. **ES1/ES10 diagnostics** (lightweight, can run early to sharpen attribution).  
7. **ES2/ES3 representation sanity checks** (deferred; current evidence says lower marginal value).  
8. **ES9 SSL/augmentation** (high-cost, only after supervised lines plateau).

## Why this order

1. **Causal clarity:** ES4 most directly tests current leading bottleneck (model-family boundary capacity) while preserving all frozen constraints.
2. **Cost/value:** ES4/ES5 are lower engineering cost than embeddings/SSL and produce immediate go/no-go evidence.
3. **Decision leverage:** ES7+ES8 are required to convert model gains into defensible promotion decisions.
4. **Avoid repeated low-yield loops:** ES2/ES3 are intentionally deprioritized to prevent another feature micro-iteration cycle.

## Deferred/high-risk sets

- **Deferred:** ES9 until supervised paths are exhausted.
- **High-risk/low-value right now:** broad handcrafted feature sweeps and detector patching.
- **Prerequisite-gated:** ES6 should be run after ES4/ES5 baseline is established for fair incremental attribution.

---

## Part E — External research and community patterns (design-relevant)

External guidance is used here only to sharpen experiment design, not to override repo evidence.

1. **GBDT for tabular interactions** (LightGBM/XGBoost docs/papers): supports ES4 as lowest-friction model-family upgrade for structured features.
2. **Ranking-aware objectives** (XGBoost LTR docs): aligns directly with observed AUC/ordering bottleneck, motivating ES5.
3. **Pretrained embeddings for low-label transfer** (OpenL3 docs, PANNs paper): motivates ES6 as a coherent representation jump when handcrafted features plateau.
4. **Calibration reliability practices** (scikit-learn calibration docs): motivates ES7 to separate ranking quality from probability usability.
5. **Uncertainty for ROC comparisons** (DeLong et al., 1988): motivates ES8 so near-threshold decisions are statistically defensible.
6. **DCASE heterogeneous-label practice** (DCASE challenge framing): supports disciplined handling of mixed annotation quality, leakage, and governance in ES8/ES9.

---

## Decision framework (promotion/stop rules)

## Global next-cycle success criteria

A candidate path is cycle-successful only if all are met:

1. Maintains frozen constraints (detector/taxonomy/labels/springboard policy).
2. Improves ranking and practical decision behavior on frozen holdout.
3. Produces at least one guardrail-feasible operating policy (or governance-approved near-pass policy with uncertainty bounds).
4. Is reproducible with explicit artifacts and row-level provenance.

## Promotion classes

- **Promote to Phase 5 r7 integration candidate:** ES4/ES5/ES6 + ES7 pass.
- **Near-pass documentation:** technical gains but no feasible operating point; route to ES8 governance decision.
- **Kill path:** fails defined success signal or cannot provide stable evidence.

---

## Risks and anti-patterns to avoid

1. Reintroducing ad hoc one-feature tweaks as “progress.”
2. Tuning on frozen scored holdout.
3. Quietly changing detector behavior while claiming classifier-only work.
4. Treating tiny point-metric differences as decisive without uncertainty.
5. Mixing track-level and global acceptance logic without explicit governance rule.
6. Reopening solved springboard line without new contradicting evidence.

---

## Part F — Concrete next-cycle recommendation

## Single best next-cycle experiment family

**Run ES4 (stronger tabular model-family benchmark) first, on frozen r4 representation and frozen data policy.**

Why: it is the shortest path to testing the strongest unresolved hypothesis (current boundary/model-family bottleneck) with high causal clarity and low implementation risk.

## Best order after that

1. ES4 -> 2. ES5 -> 3. ES7 -> 4. ES8 -> 5. ES6 (if needed) -> 6. ES9 only if still unresolved.

## Explicitly stop doing

- One-off micro-tweak loops (single scalar features, threshold-only retries) without family-level benchmark framing.
- Uncertainty-blind pass/fail calls on near-threshold deltas.

## Avoid confusing as progress

- AUC-only gains that cannot produce feasible operating behavior.
- Better threshold snapshots derived by holdout tuning.

## Evidence required to declare next cycle successful

1. Model/path candidate beats current logistic baseline with stable uncertainty-aware evidence.
2. Guardrail-feasible operating point exists under frozen constraints **or** governance explicitly accepts near-pass with predefined policy.
3. Full artifact set exists and is reproducible by future agents without hidden context.

