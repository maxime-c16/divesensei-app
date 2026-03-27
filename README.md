# DiveSensei

DiveSensei is a local-first desktop review tool for diving sessions.

The product flow is:

1. analyze a source video
2. open the session in review as soon as attempts are detected
3. review short virtual clips on the session review video
4. save `keep`, `reject`, or `unsure`
5. export only the attempts marked `keep`

This repository is the canonical product repo. The older `/home/mcauchy/divesensei` workspace is reference-only.

## What The App Does

- Astro desktop app with an Electron shell scaffold
- local SQLite catalog for sessions and review decisions via `better-sqlite3`
- review-first workflow with source-backed virtual clips
- browser-safe session review proxy generated locally
- optional export of approved clips after review
- audio-first detector variants with benchmark coverage
- local storage model that references source videos in place by default

## Repository Layout

```text
apps/desktop/   Astro app, API routes, Electron shell
src/divesensei/
  app/          CLI workflows, session pipeline, regression gates
  detection/    detector logic and audio model path
  io/           ffmpeg, OpenCV, media probing
  metadata/     UI manifest generation
  workflows/    labeling and training utilities
benchmarks/     regression manifests and comparison outputs
docs/           architecture, development, workflow notes
```

## Setup

Python runtime:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Desktop app:

```bash
cd apps/desktop
bun install
```

Or use the repo helpers:

```bash
make venv
make install
make desktop-setup
```

Required runtime dependency:

- `ffmpeg` on `PATH`
- Node.js present locally for the Astro preview server and native module rebuilds

## Main Commands

CLI:

```bash
divesensei detect /path/to/session.mov --profile long-session
divesensei detect /path/to/session.mov --profile long-session --detector-id audio_v2_pcen_classifier
divesensei inspect ./outputs/session/ui_session_manifest.json
divesensei label-audio --help
divesensei train-audio-clip-model
divesensei validate ./benchmarks/manifests/img_8237_compare.json
divesensei validate ./benchmarks/manifests/reviewed_compare.json
divesensei validate ./benchmarks/manifests/long_session_compare.json
divesensei-regress
```

Desktop:

```bash
make up
make status
make logs
make down
make re
```

Desktop dev helpers:

```bash
cd apps/desktop
bun run dev
bun run check
bun run build
bun run electron:dev
bun run electron:prepare
```

Make targets:

```bash
make desktop-check
make desktop-build
make electron-dev
make electron-prepare
```

The default demo/runtime path is `make up`, which builds the Astro app and runs `astro preview` in the background on `http://127.0.0.1:5173`.

## Detector Variants

- `audio_v1_heuristic`
  - baseline detector kept for regression and comparison
- `audio_v2_pcen_classifier`
  - current default
  - high-recall audio proposals plus short-window classifier
- `audio_v2_hybrid_video`
  - advanced audio path with optional video confirmation for ambiguous cases

## Product Notes

- review does not wait on per-attempt clip rendering
- `ready_proxy_pending` sessions are reviewable before the full session proxy finishes
- review decisions are stored locally in SQLite
- exports are secondary artifacts, not part of the primary review loop
- source videos are not duplicated by default
- the analysis form accepts either:
  - a manual absolute video path
  - a picked local file, which is copied into `.divesensei-runtime/imports/` before analysis

## Demo Branch Notes

- current demo branch: `feat/better-sqlite3-catalog`
- preferred operator flow:
  1. `make up`
  2. open `http://127.0.0.1:5173/`
  3. use the file picker or manual path
  4. review on the same branch without switching runtimes mid-demo
- use `make logs` if the UI stalls or an API route fails

## Docs

- [Architecture](docs/architecture.md)
- [Development](docs/development.md)
- [Review-First Workflow](docs/review-first-workflow.md)
- [UI Contract](docs/ui-contract.md)
- [Content Style Guide](docs/content-style-guide.md)
- [Release Guide](docs/release.md)

## Status

This repo is the product source of truth. Product and engine work should land here, not in the legacy integration workspace.
