#!/usr/bin/env bash
# Setup the YGOAGENT venv and game assets in WSL.
# Run from WSL: bash /mnt/c/Users/sungj/Desktop/Sample/ygo-agent/scripts/setup_wsl.sh
set -e

VENDOR="/mnt/c/Users/sungj/Desktop/Sample/ygo-agent/vendor/ygo-agent"
VENV="$HOME/.ygoagent_venv"

echo "=== Creating Python venv at $VENV ==="
python3 -m venv "$VENV"
source "$VENV/bin/activate"

echo "=== Installing JAX (CPU) and core deps ==="
pip install -U pip
pip install -U "jax<=0.4.28" "jaxlib<=0.4.28"
pip install flax distrax chex

echo "=== Downloading pre-built ygoenv binary (cp312) ==="
SO_DEST="$VENDOR/ygoenv/ygoenv/ygopro/ygopro_ygoenv.so"
if [ ! -f "$SO_DEST" ]; then
    wget -q "https://github.com/sbl1996/ygo-agent/releases/download/v0.1/ygopro_ygoenv_cp312.so" \
         -O "$SO_DEST"
    echo "  Downloaded to $SO_DEST"
else
    echo "  Already exists, skipping."
fi

echo "=== Installing ygoenv, ygoinf, ygoai ==="
pip install -e "$VENDOR/ygoenv"
# ygoinf lists tflite-runtime which has no cp312 wheel; install deps manually then install --no-deps
pip install numpy==1.26.4 optree fastapi "uvicorn[standard]" pydantic_settings pydantic
pip install -e "$VENDOR/ygoinf" --no-deps
pip install -e "$VENDOR"

echo "=== Downloading card database (cards.cdb, strings.conf) ==="
DB_BASE="https://github.com/mycard/ygopro-database/raw/7b1874301fc1aa52bd60585589f771e372ff52cc/locales"
mkdir -p "$VENDOR/assets/locale/en"
if [ ! -f "$VENDOR/assets/locale/en/cards.cdb" ]; then
    wget -q "$DB_BASE/en-US/cards.cdb" -O "$VENDOR/assets/locale/en/cards.cdb"
    echo "  Downloaded cards.cdb"
else
    echo "  cards.cdb already exists."
fi
if [ ! -f "$VENDOR/assets/locale/en/strings.conf" ]; then
    wget -q "$DB_BASE/en-US/strings.conf" -O "$VENDOR/assets/locale/en/strings.conf"
    echo "  Downloaded strings.conf"
else
    echo "  strings.conf already exists."
fi

echo "=== Cloning ygopro-scripts (card scripts for game engine) ==="
# Clone alongside vendor/ygo-agent/ as vendor/ygopro-scripts/ (Windows filesystem).
# The ygopro-core binary reads scripts relative to CWD using std::ifstream — keeping
# the scripts on the same filesystem as the binary avoids cross-fs symlink issues.
SCRIPTS_DIR="$(dirname "$VENDOR")/ygopro-scripts"
if [ ! -d "$SCRIPTS_DIR" ]; then
    git clone --quiet https://github.com/mycard/ygopro-scripts.git "$SCRIPTS_DIR"
fi
cd "$SCRIPTS_DIR" && git checkout --quiet 8e7fde9bf
# Relative symlink: vendor/ygo-agent/scripts/script -> ../../ygopro-scripts
LINK="$VENDOR/scripts/script"
rm -f "$LINK"
ln -sf "../../ygopro-scripts" "$LINK"
echo "  Linked $LINK -> ../../ygopro-scripts (= $SCRIPTS_DIR)"
cd "$VENDOR"

echo "=== Verifying installation ==="
python3 -c "import ygoenv; import ygoinf; import ygoai; print('  All imports OK')"

echo ""
echo "=== Done! ==="
echo "Add this to your shell profile or .env:"
echo "  YGOAGENT_VENV=$VENV"
