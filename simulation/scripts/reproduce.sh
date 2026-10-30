#!/usr/bin/env bash
# Regenerate every table and number from the archived programs (no agent calls).
#   bash scripts/reproduce.sh
set -euo pipefail
PKG="$(cd "$(dirname "$0")/.." && pwd)"
PY="${ADR_WS:-$HOME/ws}/env/bin/python"
"$PY" "$PKG/scripts/score.py"
"$PY" "$PKG/scripts/score_exact.py"
"$PY" "$PKG/scripts/make_tables.py"
