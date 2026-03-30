# Mobile V1 Shape

This document captures the concrete v1 mobile shape added to the repository for the `Capacitor shell + native iOS media engine` direction.

## Files Added

- `apps/mobile/src/native/types.ts`
- `ios/App/App/Plugins/DiveSenseiMedia/DiveSenseiMediaPlugin.swift`

## V1 Boundary

Web owns:

- session list rendering
- review timeline and queue rendering
- decision UI
- export progress UI
- player adapter selection using `ReviewPlayerAdapter`
- manifest consumption

Native owns:

- source picking and repair
- source resolution and availability
- session persistence
- analysis jobs
- manifest generation
- review proxy generation
- export rendering
- Photos save-back

## Persistence Shape

Recommended native durable entities:

1. `sources`
2. `sessions`
3. `detections`
4. `decisions`
5. `jobs`
6. `review_proxies`
7. `exports`

Durable artifacts:

- manifest JSON
- source/session/job/export metadata
- review decisions

Cache-only artifacts:

- review proxy files
- local export files

## Playback Default

Preferred default:

- `WKWebView + HTML video` using a native-generated proxy URL

Fallback:

- `native AVPlayer` behind the `ReviewPlayerAdapter` seam

Escalation:

- full native review player only if the hybrid player fails the performance spikes

## Session States In V1

- `created`
- `analyzing`
- `review_pending`
- `review_ready`
- `exporting`
- `complete_with_errors`
- `failed`
- `deleted`

## Job Phases In V1

- `source_resolving`
- `source_downloading`
- `audio_decode`
- `detecting`
- `manifest_writing`
- `proxy_generating`
- `review_ready`
- `export_preparing`
- `exporting`
- `saving_to_library`
- `completed`
- `failed`
- `cancelled`

## Source Model

Rules:

- `sourceRef` is opaque.
- No web-exposed filesystem paths.
- Photos-backed and Files-backed sources are both supported at the native layer.
- Missing and iCloud-only states are represented through `SourceAvailability`.

## Bridge Rules

Never cross the JS/native bridge:

- raw source video bytes
- raw export bytes
- native filesystem paths
- large frame buffers
- decoded audio buffers

May cross:

- opaque ids
- manifests
- lightweight URLs
- decisions
- progress payloads
- export records
