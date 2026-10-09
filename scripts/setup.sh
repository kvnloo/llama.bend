#!/usr/bin/env bash
# Checks the pinned Bend and creates the Python venv of reference/.
# Bend itself is installed by bend-ml's scripts/setup.sh (Bend 2.0.35, SHA256-checked, and Lean 4.34.0).
set -euo pipefail
cd "$(dirname "$0")/.."
BEND_VERSION=2.0.35
v=$("$HOME/.bend/bin/bend" version 2>/dev/null | awk '{print $NF}' || true)
if [ "$v" != "$BEND_VERSION" ]; then
  echo "bend $BEND_VERSION is required (found ${v:-none}); run scripts/setup.sh in bend-ml" >&2
  exit 1
fi
[ -d reference/.venv ] || python3 -m venv reference/.venv
reference/.venv/bin/pip install -q -r reference/requirements.txt
echo "setup ok: bend $v, reference/.venv"
