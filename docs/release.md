# Release Guide

## Before Release

- run `divesensei-regress`
- verify CLI help and docs
- verify `ui_session_manifest.json` and `ui_library_index.json`
- confirm no accidental benchmark artifacts are committed
- confirm no local runtime or demo-only files are committed:
  - `.run/`
  - `.divesensei-runtime/imports/`
  - local sample videos
- verify GitHub issue templates and PR template are current
- verify README matches the shipped CLI and repo layout
- verify `make up`, `make re`, and `make down` still work on a clean checkout

## Release Notes Should Include

- detector changes
- benchmark deltas
- UI contract changes
- migration notes for the desktop shell if needed
- catalog/storage migration notes when SQLite behavior changes
- deployment/runtime notes when the local demo flow changes

## Repository Hygiene

Recommended baseline before publishing:

- README present
- CONTRIBUTING guide present
- SECURITY policy present
- issue templates present
- PR template present
- release-facing docs grouped under `docs/`

Official GitHub references:

- Creating repositories with README, license, and gitignore: https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository
- About READMEs: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes
- About issue and PR templates: https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests
- GitHub Pages docs: https://docs.github.com/en/pages

## GitHub Pages

This repo includes a Pages deployment workflow in:

- [`.github/workflows/pages.yml`](/home/mcauchy/divesensei-app/.github/workflows/pages.yml)

The docs source lives in:

- [`docs/`](/home/mcauchy/divesensei-app/docs)

Before enabling Pages publicly:

- verify repository URLs
- verify docs do not reference private paths
- verify the license and visibility posture are correct
