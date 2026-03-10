# DiveSensei

Production-ready audio-first dive detection, extraction, and metadata generation for desktop and mobile companion apps.

## Product Scope

DiveSensei processes long diving-session videos and produces:

- one clip per detected dive candidate
- structured session metadata
- UI-ready manifests for future desktop apps
- benchmark and regression artifacts for detector quality control

The current production path is audio-first. Video verification remains optional.

## Repository Structure

```text
src/divesensei/
  app/         CLI-facing workflows and regression gates
  detection/   detector, config, and model hooks
  io/          ffmpeg / OpenCV / logging helpers
  metadata/    UI manifests and library indexing
  workflows/   review and training utilities
benchmarks/
  manifests/   reviewed and long-session benchmark manifests
docs/          architecture, UI contract, and release docs
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Or use the bundled bootstrap flow:

```bash
make venv
make install
make smoke-help
```

Required runtime dependency:

- `ffmpeg` on `PATH`

## CLI

```bash
divesensei detect /path/to/session.mov --profile long-session
divesensei inspect ./outputs/session/ui_session_manifest.json
divesensei validate ./benchmarks/manifests/reviewed_audio.json
divesensei review-template ./outputs/session/session_pipeline_report.json ./outputs/session/review.csv
divesensei library-index ./outputs
divesensei-regress
```

## GitHub Setup

- default CI is in [`.github/workflows/ci.yml`](/home/mcauchy/divesensei-app/.github/workflows/ci.yml)
- docs publishing is in [`.github/workflows/pages.yml`](/home/mcauchy/divesensei-app/.github/workflows/pages.yml)
- issue templates and PR template are already included under [`.github/`](/home/mcauchy/divesensei-app/.github)

Before opening the repo publicly:

1. set the real GitHub URLs in [pyproject.toml](/home/mcauchy/divesensei-app/pyproject.toml)
2. choose whether the proprietary [LICENSE](/home/mcauchy/divesensei-app/LICENSE) should stay or be replaced
3. enable GitHub Pages if you want the docs site live
4. configure repository secrets or environments only if future release workflows need them

## Docs

- [Architecture](docs/architecture.md)
- [UI Contract](docs/ui-contract.md)
- [Development Guide](docs/development.md)
- [Release Guide](docs/release.md)

## Status

This repo is the clean production base extracted from the earlier research/development workspace. Legacy experiment files are intentionally not included here.
