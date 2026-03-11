# Review-First Workflow

## Goal

The product should let a coach start reviewing immediately after detection.

The review loop is:

1. analyze the source video
2. open the session in Review as soon as attempts are written
3. review short virtual clips on the session review video
4. save `keep`, `reject`, or `unsure`
5. export only the kept attempts

## Core Principles

- review is the primary workflow
- exports are secondary
- source videos stay referenced in place by default
- per-attempt rendering is not required before review starts
- queue decisions are saved continuously

## Review Surface

The Review tab is the working surface.

- left pane
  - session review video
  - bounded virtual playback
  - previous and next
  - play, loop, autoplay, auto-next
  - current decision and optional note
  - timeline of detected attempts
- right pane
  - compact attempt queue
  - plain review terms instead of diagnostic jargon
  - filters by decision state

## Virtual Clips

The primary review mode is source-backed.

- no wait for extracted clips
- each attempt uses a short review window around the detected splash
- the user reviews attempt-sized segments on the session review video

Current review window:

- `2s` before splash
- `2s` after splash

## Exports

Exports happen after review.

- only attempts marked `keep` are rendered
- exports are durable clip files
- exports are not part of the normal analysis wait path

Current export window:

- `6s` before splash
- `3s` after splash

## Session Readiness

The session can be reviewed before the full review proxy finishes.

- `ready_proxy_pending`
  - attempts are available
  - review opens immediately
  - full review proxy may still be rendering
- `complete`
  - review proxy is ready

## What This Replaced

This flow replaces the old extraction-first pattern where every candidate clip had to be rendered before the review workspace felt usable.

That older model increased latency, disk use, and background processing for attempts the user might reject anyway.
