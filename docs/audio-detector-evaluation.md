# Audio Detector Evaluation

## Core Architecture

DiveSensei stays on this layered detector path:

1. source video or cached audio
2. mono 16 kHz audio extraction
3. mel-like multiband frontend plus PCEN onset features
4. high-recall proposal detection
5. candidate window feature extraction
6. engineered clip features
7. logistic-regression clip classifier
8. thresholding / suppression / ambiguity handling
9. review workflow and retraining loop

The detector family is intentionally inspectable:

- no CNN replacement
- no opaque end-to-end audio model in the main path
- reviewed sessions are the center of iteration

## Session Concepts

- `candidate`
  - a detector-selected final attempt shown in the UI review queue
- `reviewed candidate`
  - a candidate with a human decision such as `dive`, `non_dive`, or `unsure`
- `false negative`
  - a manually logged missed dive timestamp, not attached to an existing candidate by default
- `mined hard negative`
  - a reviewed `non_dive` candidate exported for future training
- `training example`
  - a saved labeled audio clip produced by `label-audio`

False negatives are separate manual annotations because a missed dive can happen when no final candidate exists at all.

## Acoustic Intuition

What we want:

- true dive entry splashes
  - broadband, turbulent water-entry energy
  - strong post-entry tail compared with a dry impact

Main confusers:

- board rebound / board slap
  - often sharper and more resonant
  - can produce strong transient peaks with ringing structure
- non-dive splashes
  - splash-like but often shorter or differently shaped in decay
- voice / whistle / applause / deck handling noise
  - can create narrowband or impulsive false positives

Why they are confusable:

- all of them can generate high transient intensity
- pool acoustics and mic placement change spectral balance a lot
- board rebounds can look splash-like if the model overweights raw impulse strength

## Current Strengths

- audio-first evaluation sessions avoid repeated full-video ingest
- cached WAV reuse makes detector iteration fast
- reviewed `non_dive` decisions now export directly into hard-negative mining artifacts
- reviewed false negatives can be attributed to detector stages

## Current Limits

- pool and mic domain shift remains a major source of instability
- classifier calibration still depends on having enough reviewed candidates with real probabilities
- small local validation sets can overfit logistic weights very easily
- hard-negative subtypes improve analysis, but subtype coverage still depends on human review discipline

## Audio Assumptions

- sample rate: 16 kHz
- channel layout: mono
- clip labeling uses PCM 16-bit WAV
- detector timestamps are centered on audio proposals and then converted into review windows
- feature extraction assumes consistent timing alignment between cached audio and the source proxy video
- normalization is feature-level z-scoring inside the logistic model, not global loudness normalization across the whole dataset
