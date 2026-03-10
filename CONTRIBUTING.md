# Contributing

## Principles

- keep the reviewed benchmark as the non-regression gate
- keep the long-session benchmark as the product realism gate
- prefer small, testable changes over large detector rewrites
- preserve stable UI manifests when changing backend internals

## Development Flow

1. create a branch
2. run local smoke checks
3. run `divesensei-regress`
4. update docs if the CLI, metadata contract, or architecture changes

## Pull Requests

Each PR should state:

- what changed
- why it changed
- benchmark impact
- UI contract impact

