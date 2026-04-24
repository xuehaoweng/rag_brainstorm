#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail

cd "$(dirname "$0")/.."

pids=()

cleanup() {
  if ((${#pids[@]} > 0)); then
    kill "${pids[@]}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

printf 'Starting Self RAG development stack...\n'
printf '  Backend:  http://0.0.0.0:8800\n'
printf '  Frontend: http://0.0.0.0:5173\n\n'

bash scripts/dev-backend.sh &
pids+=("$!")

bash scripts/dev-frontend.sh &
pids+=("$!")

wait -n "${pids[@]}"
