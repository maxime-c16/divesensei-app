# UI Contract

This repo is structured so a future Electron, Tauri, or mobile shell can be built on top of stable files rather than internal Python objects.

## Core Principle

The frontend should render persisted metadata.

The backend should own:

- media processing
- detection
- extraction
- manifest generation
- library indexing

The frontend should own:

- presentation
- user interaction
- local UI state
- navigation

## Canonical Artifacts

Every session run should produce:

- `session_pipeline_report.json`
  - full backend report
- `ui_session_manifest.json`
  - canonical UI-facing session object
- `session_debug_summary.json`
  - compact operational summary
- `session_pipeline.log.jsonl`
  - append-only structured run log
- `detections.csv`
  - tabular export

For collection views:

- `ui_library_index.json`
  - one index built by scanning many session manifests

## Why This Contract Exists

Desktop UI code should not need to:

- parse raw detector internals
- guess clip naming rules
- derive list-view metadata from logs
- reach into Python process memory

Instead it should:

- open `ui_library_index.json` for the home/library screen
- open `ui_session_manifest.json` for the session detail screen
- open linked artifacts only when the user asks for debug detail

## ui_session_manifest.json

Top-level sections:

- `schema_version`
- `kind`
- `generated_at`
- `session`
- `artifacts`
- `detections`

### session

Use this for:

- title/header
- run status
- profile badge
- telemetry cards
- counts and summary

### artifacts

Use this for:

- “open raw report”
- “download CSV”
- “open logs”
- “debug panel”

### detections

Use this for:

- timeline markers
- table rows
- detail drawers
- clip browser cards

Each detection includes:

- stable id
- timestamps
- duration
- confidence
- scores
- feature values
- clip path metadata

## ui_library_index.json

Use this for:

- session list pages
- dashboard tiles
- search/filter results
- recent runs

The UI should prefer the library index for overview screens and load a full session manifest only when the user drills into a session.

## Recommended Desktop Bridge

### Electron

Recommended shape:

- main process
  - starts backend jobs
  - owns filesystem access
  - watches output directories
- preload
  - exposes a narrow API through `contextBridge`
- renderer
  - renders library/session pages
  - consumes only sanitized data

Use:

- `contextIsolation`
- narrow IPC
- least-privilege renderer surface

Official references:

- Process model: https://www.electronjs.org/docs/latest/tutorial/process-model
- Context isolation: https://www.electronjs.org/docs/latest/tutorial/context-isolation
- IPC: https://www.electronjs.org/docs/latest/tutorial/ipc
- Security: https://www.electronjs.org/docs/latest/tutorial/security

### Tauri

Recommended shape:

- Rust commands for backend actions
- events for progress
- persisted JSON files for completed run state
- optional store only for lightweight UI preferences

Use commands for:

- run detection
- inspect session
- rebuild library index
- open output directory or clip

Official references:

- Calling Rust commands: https://v2.tauri.app/develop/calling-rust/
- State management: https://v2.tauri.app/develop/state-management/
- Store plugin: https://v2.tauri.app/plugin/store/

## Suggested UI Sections

### Library

- cards from `ui_library_index.json`

### Session Overview

- source video
- profile
- counts
- telemetry
- artifact links

### Timeline

- ordered detection markers using `timestamp_seconds`

### Detection Table

- index
- timestamp
- duration
- confidence
- audio score
- combined score

### Detection Detail

- clip preview
- score breakdown
- feature values
- raw artifact links

### Debug

- session debug summary
- JSONL log viewer
- raw report link

## Versioning Rule

If UI-facing fields change:

- update `schema_version`
- document the change
- keep the desktop app backward-compatible when possible

