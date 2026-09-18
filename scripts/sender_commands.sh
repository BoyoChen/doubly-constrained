#!/usr/bin/env bash
# Run from any directory on the Linux training host. Default only prints a plan.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
action="${1:-plan}"
label="${2:-paper-v3}"
case "$action" in
  plan) python "$here/run_suite.py" sender --label "$label" ;;
  train)
    python "$here/run_suite.py" sender --stage prefix --label "$label" --execute
    python "$here/run_suite.py" sender --stage continuation --label "$label" --execute
    python "$here/sender_results.py" --label "$label"
    ;;
  analyze) python "$here/sender_results.py" --label "$label" ;;
  *) echo 'Usage: bash sender_commands.sh {plan|train|analyze} [run-label]' >&2; exit 2 ;;
esac
