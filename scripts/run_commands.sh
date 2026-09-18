#!/usr/bin/env bash
# Activate boyonet first. No Prefect server or historical code checkout required.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
action="${1:-plan}"
label="${2:-paper-v3}"
case "$action" in
  plan) python "$here/run_suite.py" all --label "$label" ;;
  check)
    python "$here/validate.py" --construct-models
    python "$here/smoke.py"
    python "$here/artifact_smoke.py"
    ;;
  train)
    python "$here/run_suite.py" all --label "$label" --execute
    python "$here/chapter_results.py" --label "$label"
    ;;
  analyze) python "$here/chapter_results.py" --label "$label" ;;
  *) echo 'Usage: bash run_commands.sh {plan|check|train|analyze} [run-label]' >&2; exit 2 ;;
esac
