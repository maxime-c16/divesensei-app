# Architecture

DiveSensei has two product layers:

1. Python analysis engine in `src/divesensei/`
2. Astro desktop app in `apps/desktop/`

The desktop app is the operator surface. The Python layer owns detection, manifests, exports, and regression tooling.

## Runtime Model

The app is local-first.

- source videos stay in place by default
- generated artifacts are written under `outputs/`
- desktop runtime state lives under `.divesensei-runtime/`
- session catalog and review decisions are stored in local SQLite

The desktop library should treat the catalog as the source of truth for known sessions.

## Review-First Pipeline

Current product flow:

1. detect attempts from the source video
2. write the UI session manifest immediately
3. mark the session reviewable as soon as attempts exist
4. generate the browser-safe session review proxy in the background
5. review attempts as bounded virtual clips on the session review video
6. export real clip files only for attempts marked `keep`

This is intentionally different from the old extraction-first flow.

## Detection Stack

Detector variants:

- `audio_v1_heuristic`
  - baseline detector kept for comparison
- `audio_v2_pcen_classifier`
  - default path
  - high-recall audio proposals plus short-window classifier
- `audio_v2_hybrid_video`
  - audio path with optional video confirmation for ambiguous cases

The baseline stays available so benchmark results remain comparable across iterations.

## Main Artifacts

- `ui_session_manifest.json`
  - the desktop app contract for one session
- `session_pipeline_report.json`
  - detailed pipeline output
- `session_pipeline.log.jsonl`
  - structured progress log
- `web/session_source_review.mp4`
  - browser-safe session review video
- `exports/...`
  - approved clip exports created after review

## Session States

- `ready_proxy_pending`
  - attempts are ready for review
  - full review proxy is still being prepared
- `complete`
  - session review proxy is ready
- `complete_with_errors`
  - session exists but some downstream artifact work failed
- `complete_proxy_error`
  - review remains available even if proxy generation failed

## Benchmarks And Regression

Regression is part of the product, not an afterthought.

- benchmark manifests live under `benchmarks/manifests/`
- baseline and advanced detectors are compared side by side
- detector changes should be validated before defaults move

## Desktop App Responsibilities

`apps/desktop` owns:

- session library
- analysis launcher
- review queue
- review decisions
- export jobs
- local media APIs
- Electron packaging path

It should not invent its own data model separate from the manifest and catalog layer.
