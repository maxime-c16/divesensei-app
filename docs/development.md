# Development

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Or:

```bash
make venv
make install
```

## Main Commands

```bash
divesensei detect /path/to/session.mov --profile long-session
divesensei validate ./benchmarks/manifests/reviewed_audio.json
divesensei-regress
make compile
make smoke-help
```

## Model Iteration

1. run a session
2. create a review template
3. label clips
4. export features
5. train a candidate model
6. test it behind the regression gate
