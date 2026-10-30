#!/usr/bin/env bash
# Regenerate every number of this study from the archived programs and baseline outputs (no agent call).
#   bash scripts/setup.sh && bash scripts/reproduce.sh
set -euo pipefail
PKG="$(cd "$(dirname "$0")/.." && pwd)"
PY="${ADR_WS:-$HOME/ws}/env/bin/python"
cd "$PKG"
python3 scripts/make_dequantized.py --check
"$PY" scripts/score.py
"$PY" scripts/make_tables.py
