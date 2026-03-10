# Architecture

DiveSensei is organized around a simple production path:

1. detect audio-driven dive candidates
2. export clips and metadata
3. persist UI-ready manifests
4. guard changes with regression benchmarks

## Layers

- `app`
  - user-facing workflows
- `detection`
  - detector logic and scoring hooks
- `io`
  - ffmpeg, OpenCV, logging
- `metadata`
  - stable manifests for UI integration
- `workflows`
  - operator review and model-training steps

## Source Of Truth

- backend reports for raw detail
- UI manifests for app consumption
- benchmark manifests for regression control

