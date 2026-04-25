# R37 Approve Safety Monitor Workflow

This workflow makes the approve/review safety monitor part of the normal review/export loop.

## Command

Run after reviewing and exporting a session:

```bash
make approve-safety-monitor
```

The command refreshes the research monitor and publishes stable latest-status artifacts:

- `outputs/latest_approve_safety_monitor.json`
- `outputs/latest_approve_safety_monitor.md`

## Operating Rule

1. Review the session in the desktop UI.
2. Export reviewed labels/artifacts.
3. Run `make approve-safety-monitor`.
4. If dangerous approvals are `0`, do not run policy search and do not promote the visual veto.
5. If dangerous approvals are greater than `0`, open a focused hard-negative diagnosis pass for the new source family.

## What Not To Do When Clean

- Do not lower approval thresholds.
- Do not promote the visual veto from calibration-only evidence.
- Do not run broad approve-policy sweeps.
- Do not work on auto-exclude.

## Acquisition Targets

New videos are available under `/Users/mcauchy/Library/Mobile Documents/com~apple~CloudDocs/Diving/Training`. Use them as future reviewed sources when the goal is hard-negative acquisition. Prioritize sessions with:

- shammy/towel throws in platform context
- handling noise near platform
- close-mic whistle or voice
- non-dive splash in platform context
- phone/camera movement near impact
- pool-deck impact-like transients

## Current Status

- rows considered: `445`
- newly included extension rows: `24`
- v1 approvals: `38`
- dangerous approvals: `0`
- hard-negative diagnosis trigger: `False`

## Decisions

- `R37_APPROVE_SAFETY_MONITOR_OPERATIONALIZED`
- `APPROVE_REVIEW_V1_REMAINS_DEFAULT`
