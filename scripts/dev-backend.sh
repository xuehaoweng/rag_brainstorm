#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail

cd "$(dirname "$0")/.."
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8800
