# Desktop Migration Plan

`/home/mcauchy/divesensei-app` is the canonical repo.

This document records the audit against the old integration workspace and the concrete migration path to bring the current real product into this repo.

## Audit Summary

### Present in `divesensei-app`

- packaged Python analysis engine under `src/divesensei/`
- benchmark manifests and regression-oriented docs
- no current Astro app source
- no current Electron shell source
- no local catalog runtime
- no local analysis launcher from the UI

### Present in old workspace and still required

- single-design MEET review workspace
- Astro app with local review APIs
- local analysis launcher
- local SQLite catalog
- review-ready sessions before full proxy completion via `ready_proxy_pending`
- source video plus extracted clip review
- Electron preload/main scaffold
- unsigned macOS Intel DMG packaging path
- local-first storage model with source-video references instead of duplication

### Product-Critical Gaps To Close

1. `divesensei-app` Python pipeline is behind the real runtime behavior:
   - default extraction window differs from the active workspace
   - no review proxy generation
   - no early manifest write with `ready_proxy_pending`
2. `divesensei-app` has no canonical desktop app package:
   - no Astro route tree
   - no local media/catalog APIs
   - no Electron bridge
3. current repo docs still describe only the engine, not the canonical app+engine split

## Target Repo Shape

```text
divesensei-app/
├── src/divesensei/        # canonical Python engine
├── apps/desktop/          # canonical Astro + Electron desktop app
├── docs/                  # architecture, UI contract, migration status
├── benchmarks/            # regression inputs
└── Makefile               # root bootstrap and smoke targets
```

## Migration Phases

### Phase 1: Align engine behavior with shipped UI expectations

- port review proxy generation into `src/divesensei/io/media_io.py`
- update `src/divesensei/app/session_pipeline.py`
  - extraction default `post = 3.0`
  - write manifest/report before full review proxy completes
  - emit `ready_proxy_pending`
  - rewrite manifest to `complete` or `complete_proxy_error` after proxy completion
- keep canonical artifacts and schema stable

### Phase 2: Seed canonical desktop app in `apps/desktop`

- port the single-design MEET workspace
- port local runtime config, session catalog, media API, and analysis launcher
- point the app at the canonical Python package, not the old workspace
- preserve local-first media rules

### Phase 3: Complete desktop shell packaging

- port Electron `main`, `preload`, dev runner, packaged runner
- port packaged Astro server patching for Electron
- restore unsigned macOS Intel DMG build path under the canonical repo

### Phase 4: Remove transitional dependencies on old workspace

- eliminate hardcoded references to `/home/mcauchy/divesensei`
- make all runtime roots relative to `divesensei-app`
- document the final bootstrap and packaging commands in this repo only

## Current Implementation Slice

This migration pass starts Phase 1 and seeds Phase 2:

- engine behavior is updated first because the desktop UI depends on it
- `apps/desktop` is introduced as the canonical home for the MEET workspace and local desktop runtime
- the old workspace remains reference-only until the remaining Electron/package polish is ported
