# Benchmarks

This directory contains the manifests used as quality gates.

- `manifests/img_8237_compare.json`
  - hard timestamp regression for the horn/clapping false-positive case
- `manifests/reviewed_compare.json`
  - reviewed set comparing baseline vs advanced detector
- `manifests/long_session_compare.json`
  - real long-session gate comparing baseline vs advanced detector

Use:

```bash
divesensei validate ./benchmarks/manifests/img_8237_compare.json
divesensei validate ./benchmarks/manifests/reviewed_compare.json
divesensei validate ./benchmarks/manifests/long_session_compare.json
divesensei-regress
```
