#!/usr/bin/env bash
set -euo pipefail

KAGGLE_BIN="${KAGGLE_BIN:-kaggle}"

if ! command -v "${KAGGLE_BIN}" >/dev/null 2>&1; then
  cat >&2 <<'EOF'
Kaggle CLI is not installed.

Install it with:
  python3 -m venv .venv-kaggle
  .venv-kaggle/bin/python -m pip install --upgrade pip kaggle
  export KAGGLE_BIN=.venv-kaggle/bin/kaggle

Then authenticate with one of:
  KAGGLE_ENABLE_OAUTH=1 .venv-kaggle/bin/kaggle auth login
or:
  mkdir -p ~/.kaggle
  cp /path/to/kaggle.json ~/.kaggle/kaggle.json
  chmod 600 ~/.kaggle/kaggle.json
EOF
  exit 127
fi

echo "Kaggle CLI:"
"${KAGGLE_BIN}" --version

echo
echo "Auth check:"
"${KAGGLE_BIN}" datasets list --mine >/tmp/r41_kaggle_auth_check.txt
head -n 5 /tmp/r41_kaggle_auth_check.txt

echo
echo "Kaggle CLI authentication is working."
