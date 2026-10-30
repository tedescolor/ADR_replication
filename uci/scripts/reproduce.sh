#!/usr/bin/env bash
# Regenerate every table and number from the archived programs (no agent calls).
#   bash scripts/reproduce.sh
set -euo pipefail
PKG="$(cd "$(dirname "$0")/.." && pwd)"
PY="${ADR_WS:-$HOME/ws}/env/bin/python"
cd "$PKG"
export PYTHONDONTWRITEBYTECODE=1  # importing the archived model.py files must not leave __pycache__ in results/
"$PY" scripts/score.py
"$PY" scripts/make_tables.py
"$PY" scripts/make_usage.py
