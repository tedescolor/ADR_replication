#!/usr/bin/env bash
# Step 1 of the README: environment, data files check, library snapshot, sandbox settings.
#   bash scripts/setup.sh
# Idempotent: safe to run again. The workspace defaults to ~/ws (override with ADR_WS).
set -euo pipefail
PKG="$(cd "$(dirname "$0")/.." && pwd)"
WS="${ADR_WS:-$HOME/ws}"

say() { printf '[setup] %s\n' "$*"; }
fail() { printf '[setup] ERROR: %s\n' "$*" >&2; exit 1; }

# 1. the workspace path is shown to the agents: it must be neutral
if printf '%s' "$WS" | grep -qiE 'uci|redo|cdtree|yang|leeuwen|bench|cpfn|franco|tedesco|branched|simul|blobs|multiv'; then
  fail "workspace path '$WS' contains an identifying word; use a neutral workspace such as ADR_WS=/ws (see the top-level README)"
fi
mkdir -p "$WS/runs"
say "workspace: $WS"

# 2. tools
for t in claude bwrap socat; do
  command -v "$t" >/dev/null || fail "'$t' not found (sudo apt install bubblewrap socat)"
done
say "claude $(claude --version 2>/dev/null | head -1); bwrap and socat present"
if ! bwrap --ro-bind / / --unshare-net --dev /dev --proc /proc true 2>/dev/null; then
  fail "bubblewrap cannot create a sandbox on this machine (AppArmor user-namespace restriction).
        Ask the operator to run once:  sudo bash $PKG/scripts/fix_apparmor.sh"
fi
say "bubblewrap sandbox works"

# 3. Python environment for agents and scripts (Python 3.11)
if [ ! -x "$WS/env/bin/python" ]; then
  if command -v uv >/dev/null; then uv venv -q --python 3.11 "$WS/env"
  else python3.11 -m venv "$WS/env"; fi
fi
if command -v uv >/dev/null; then uv pip install -q --python "$WS/env/bin/python" -r "$PKG/requirements.txt"
else "$WS/env/bin/pip" install -q -r "$PKG/requirements.txt"; fi
"$WS/env/bin/python" -c "import numpy, scipy, pandas, matplotlib, ot; print('[setup] python', __import__('sys').version.split()[0], 'numpy', numpy.__version__, 'scipy', scipy.__version__)"

# 4. integrity of the data files (from the CPFN repository; blobs_3ex_true.csv regenerated with R)
(cd "$PKG/data" && sha256sum --quiet -c SHA256SUMS) || fail "data/ checksum mismatch"
say "data/: training set, test set, KCDE samples, 3-example samples: checksums OK"

# 5. frozen library given to every agent
if [ -d "$PKG/adr_snapshot" ]; then
  diff -rq -x __pycache__ "$PKG/adr" "$PKG/adr_snapshot" >/dev/null \
    || fail "adr_snapshot differs from adr: the library must not change"
else
  cp -r "$PKG/adr" "$PKG/adr_snapshot"; find "$PKG/adr_snapshot" -name __pycache__ -prune -exec rm -rf {} +
fi
say "adr_snapshot identical to adr"

# 6. sandbox settings template. Every agent gets its own file, rendered by the scripts with the
#    absolute path of its task directory (relative paths such as "." would resolve against the
#    folder of the settings file, not the task directory).
mkdir -p "$WS/settings"
sed -e "s#__WS__#$WS#g" -e "s#__HOME__#$HOME#g" "$PKG/sandbox_settings.json" > "$WS/settings.template.json"
rm -f "$WS/settings.json"
say "sandbox settings template: $WS/settings.template.json"

# 7. results tree
mkdir -p "$PKG/results"
[ -f "$PKG/results/audit_notes.md" ] || printf '# Audit notes\n\n' > "$PKG/results/audit_notes.md"
say "done. Next: $WS/env/bin/python scripts/probe.py"
