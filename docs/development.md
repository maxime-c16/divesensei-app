# Development

## Environment

Python:

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

Optional repo helpers:

```bash
make venv
make install
make desktop-setup
```

Required runtime dependency:

- `ffmpeg` on `PATH`
- Node.js available locally for the Astro server and `better-sqlite3` rebuild

## Main Commands

Engine and tooling:

```bash
PYTHONPATH=src python3 -m divesensei.cli --help
PYTHONPATH=src python3 -m divesensei.cli detect --help
divesensei detect /path/to/session.mov --profile long-session
divesensei label-audio --help
divesensei train-audio-clip-model
divesensei validate ./benchmarks/manifests/img_8237_compare.json
divesensei validate ./benchmarks/manifests/reviewed_compare.json
divesensei validate ./benchmarks/manifests/long_session_compare.json
divesensei-regress
```

Desktop app:

```bash
make up
make down
make re
make status
make logs
```

Local dev helpers:

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
make compile
make smoke-help
make desktop-check
make desktop-build
make electron-dev
make electron-prepare
```

`make up` is the stable local deployment target. It builds the Astro app if needed, starts `astro preview` in the background on `http://127.0.0.1:5173`, and stores runtime state under `.run/`.

## Working Rules

- use `/home/mcauchy/divesensei-app` as the canonical repo
- treat `/home/mcauchy/divesensei` as reference-only
- preserve the review-first product workflow
- keep `audio_v1_heuristic` benchmarkable whenever detector logic changes
- avoid duplicate source-video storage by default

## Detector Iteration

Expected loop:

1. run the target session with the current detector
2. save hard positive and hard negative audio labels
3. retrain the short-window clip model if needed
4. run benchmark manifests against baseline and advanced paths
5. only change defaults after regression stays acceptable

## Desktop Runtime Notes

- `.divesensei-runtime/`
  - SQLite catalog
  - analysis job state
  - export job state
  - labeled audio clips
  - local models
  - imported picker uploads under `imports/`
- `outputs/`
  - session manifests
  - review proxies
  - exported clips

## Review Workflow Notes

- the primary review surface uses virtual clips over the session review video
- exported clips are optional derived artifacts
- the library should reflect catalog state, not blind directory discovery
- sessions should remain reviewable when proxy generation is still pending
- the analysis launcher supports two input modes:
  - manual absolute path entry
  - file picker import, which copies the selected file into `.divesensei-runtime/imports/` before analysis starts

## Demo Notes

- for the demo branch, prefer `make re` over restarting `bun run dev`
- if a picked-file analysis appears stalled, check `.run/desktop-preview.log`
- avoid committing `.run/`, local sample videos, or generated `outputs/` artifacts unless intentionally updating fixtures
