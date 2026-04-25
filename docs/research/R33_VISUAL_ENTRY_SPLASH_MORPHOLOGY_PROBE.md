# R33 Visual Entry/Splash Morphology Probe

R33 tested a bounded visual morphology probe on the r32 hard-negative clip bank. The existing visual late-fusion score fails because it scores the shammy/non_dive_splash hard negative near 1.0. The new morphology features expose a more specific direction: real entry/splash controls and the shammy nuisance differ in motion locality, area, persistence, and waterline/flow structure.

The pass also recovered the legacy v1 splash recognition cues from the original detector: lower-water splash motion, pre-entry diver motion, splash/pre and splash/post ratios, and the legacy video score formula. These are included as explicit r33 features so the old verifier logic can be compared against the newer visual late-fusion score.

This is diagnostic only. Do not wire a product policy from 7 clips. The next bounded step is to expand the same features over all reviewed source clips or a fresh independent nuisance-heavy reviewed session.

Decisions:

- `R33_VISUAL_HARD_NEGATIVE_PROBE_DIAGNOSTIC_GAIN`
- `APPROVE_REVIEW_V1_REMAINS_DEFAULT`
