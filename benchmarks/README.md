# Benchmarks

This directory contains the manifests used as quality gates.

- `manifests/reviewed_audio.json`
  - reviewed benchmark set
- `manifests/long_session.json`
  - real long-session product gate

Use:

```bash
divesensei validate ./benchmarks/manifests/reviewed_audio.json
divesensei validate ./benchmarks/manifests/long_session.json
divesensei-regress
```
