#!/usr/bin/env bash
# Setup ygo-meta-ai on a cloud GPU instance (RunPod, Lambda, etc.).
# Assumes: NVIDIA GPU + CUDA drivers already installed (nvidia-smi works).
# Run: bash scripts/setup_cloud_gpu.sh
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR="$REPO_ROOT/vendor/ygo-agent"
VENV="${YGOAGENT_VENV:-/workspace/venv}"

echo "=== Creating Python venv at $VENV ==="
python3 -m venv "$VENV"
source "$VENV/bin/activate"

echo "=== Installing JAX with CUDA support ==="
pip install -U pip
pip install -U "jax[cuda12]<=0.4.28"
pip install flax distrax chex

echo "=== Verifying CUDA backend ==="
python3 -c "import jax; jax.local_devices(); assert jax.default_backend() == 'gpu', 'GPU not detected'; print(f'  JAX backend: {jax.default_backend()}, devices: {jax.local_devices()}')"

echo "=== Downloading pre-built ygoenv binary ==="
PY_VER=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
SO_DEST="$VENDOR/ygoenv/ygoenv/ygopro/ygopro_ygoenv.so"
if [ ! -f "$SO_DEST" ]; then
    # Only cp312 binary available upstream; check compatibility
    if [ "$PY_VER" != "cp312" ]; then
        echo "  WARNING: pre-built binary is cp312, you have $PY_VER — may not work"
    fi
    wget -q "https://github.com/sbl1996/ygo-agent/releases/download/v0.1/ygopro_ygoenv_cp312.so" \
         -O "$SO_DEST"
    echo "  Downloaded to $SO_DEST"
else
    echo "  Already exists, skipping."
fi

echo "=== Installing ygoenv, ygoinf, ygoai ==="
pip install -e "$VENDOR/ygoenv"
pip install numpy==1.26.4 optree fastapi "uvicorn[standard]" pydantic_settings pydantic
pip install -e "$VENDOR/ygoinf" --no-deps
pip install -e "$VENDOR"

echo "=== Installing ygo-meta-ai ==="
pip install -e "$REPO_ROOT"

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

echo "=== Cloning ygopro-scripts ==="
SCRIPTS_DIR="$REPO_ROOT/vendor/ygopro-scripts"
if [ ! -d "$SCRIPTS_DIR" ]; then
    git clone --quiet https://github.com/mycard/ygopro-scripts.git "$SCRIPTS_DIR"
fi
cd "$SCRIPTS_DIR" && git checkout --quiet 8e7fde9bf
LINK="$VENDOR/scripts/script"
rm -f "$LINK"
ln -sf "../../ygopro-scripts" "$LINK"
echo "  Linked $LINK -> ../../ygopro-scripts"

echo "=== Building card name DB ==="
cd "$REPO_ROOT"
python3 scripts/build_card_db.py

echo "=== Verifying installation ==="
python3 -c "import ygoenv; import ygoinf; import ygoai; print('  All imports OK')"

echo ""
echo "=== Done! ==="
echo "GPU training ready. Run:"
echo "  source $VENV/bin/activate"
echo "  python -m ygo_meta.cli.train --archetypes K9Vanquishsoul --archetypes BrandedDracotail --timesteps 5000000"
