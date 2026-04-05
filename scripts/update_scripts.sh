#!/usr/bin/env bash
# Update vendor/ygopro-scripts to a specific commit (or latest origin/master).
# Also patches the pinned hash in setup_wsl.sh so other machines stay in sync.
#
# Usage:
#   ./scripts/update_scripts.sh              # update to latest origin/master
#   ./scripts/update_scripts.sh <commit>     # update to specific commit

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS_DIR="$REPO_ROOT/vendor/ygopro-scripts"
SETUP_SH="$REPO_ROOT/scripts/setup_wsl.sh"

if [ ! -d "$SCRIPTS_DIR" ]; then
    echo "vendor/ygopro-scripts not found — run scripts/setup_wsl.sh first."
    exit 1
fi

cd "$SCRIPTS_DIR"
git fetch origin --quiet

TARGET="${1:-origin/master}"
git checkout --quiet "$TARGET"

NEW_HASH="$(git rev-parse --short HEAD)"
echo "Checked out $TARGET -> $NEW_HASH"

# Update the pinned commit in setup_wsl.sh
sed -i "s/git checkout --quiet [0-9a-f]*/git checkout --quiet $NEW_HASH/" "$SETUP_SH"
echo "Updated setup_wsl.sh pin to $NEW_HASH"
