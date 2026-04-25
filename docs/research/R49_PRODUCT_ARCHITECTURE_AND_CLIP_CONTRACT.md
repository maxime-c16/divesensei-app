# R49 Product Architecture And Clip Contract

DiveSensei should be evaluated as an audio-anchored dive clip extraction and review system, with visual recovery support.

## Decisions

- Audio anchors are primary.
- Clip extraction should use audio anchor plus configurable buffers.
- Exact visual splash/contact localization is not required.
- Visual-only recovery remains useful as a research/support branch.
- Audio-window hard verification is rejected after r48.

## Best Next Engineering Step

Implement clip extraction presets around audio anchors and add a clip-quality review signal. This is more product-aligned than more prompt hunting.
