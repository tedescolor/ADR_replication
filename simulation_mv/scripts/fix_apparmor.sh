#!/usr/bin/env bash
# Ubuntu 23.10+/24.04 blocks unprivileged user namespaces with AppArmor
# (kernel.apparmor_restrict_unprivileged_userns = 1), so bubblewrap -- and with it Claude Code's
# Bash sandbox -- cannot start ("setting up uid map: Permission denied" or
# "loopback: Failed RTM_NEWADDR"). This installs a profile that allows user namespaces for
# /usr/bin/bwrap only (the system-wide restriction stays on). Run once, with sudo:
#   sudo bash scripts/fix_apparmor.sh
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
cat > /etc/apparmor.d/bwrap <<'PROFILE'
abi <abi/4.0>,
include <tunables/global>

profile bwrap /usr/bin/bwrap flags=(unconfined) {
  userns,
  include if exists <local/bwrap>
}
PROFILE
apparmor_parser -r /etc/apparmor.d/bwrap
echo "AppArmor profile for /usr/bin/bwrap installed."
