#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$REPO_ROOT/.venv"
ENV_FILE="$REPO_ROOT/.env"
ENV_EXAMPLE="$REPO_ROOT/.env.example"

echo "=== SmartBlog LiveAvatar Worker Setup ==="
echo "Repository: $REPO_ROOT"

# ── Check Python ──
PYTHON=$(command -v python3 || true)
if [ -z "$PYTHON" ]; then
    echo "ERROR: python3 not found. Install Python 3.10+ first."
    exit 1
fi
PY_VERSION=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python: $PYTHON ($PY_VERSION)"

# ── Check .env ──
if [ ! -f "$ENV_FILE" ]; then
    echo ""
    echo "WARNING: .env not found. Copying from .env.example"
    cp "$ENV_EXAMPLE" "$ENV_FILE"
    echo "Please edit $ENV_FILE and fill in your keys before starting the worker."
fi

# ── Create venv ──
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "Creating virtual environment..."
    $PYTHON -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q

# ── Install PyTorch (CUDA 12.8) ──
echo ""
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 -q

# ── Install requirements ──
echo ""
echo "Installing requirements..."
pip install -r "$REPO_ROOT/requirements.txt" -q

# ── Patch basicsr compatibility ──
SITE_PACKAGES=$("$VENV_DIR/bin/python" -c "import site; print(site.getsitepackages()[0])")
DEGRADATIONS="$SITE_PACKAGES/basicsr/data/degradations.py"
if [ -f "$DEGRADATIONS" ] && grep -q "functional_tensor" "$DEGRADATIONS"; then
    sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' "$DEGRADATIONS"
    echo "Patched basicsr: functional_tensor -> functional"
fi

# ── Check ffmpeg / ffprobe ──
for cmd in ffmpeg ffprobe; do
    if ! command -v $cmd &>/dev/null; then
        echo "WARNING: $cmd not found. Install ffmpeg with NVENC support."
    fi
done

# ── Download checkpoints from HuggingFace ──
echo ""
echo "Checking model checkpoints..."
"$VENV_DIR/bin/python" -c "
import sys, os
sys.path.insert(0, '$REPO_ROOT')
os.chdir('$REPO_ROOT')
from pathlib import Path

# Load config and env
CONFIG_PATH = Path('$REPO_ROOT/worker_config.json')
ENV_PATH = Path('$REPO_ROOT/.env')
import json
if CONFIG_PATH.exists():
    data = json.loads(CONFIG_PATH.read_text())
    for key, value in data.items():
        if value is not None:
            os.environ.setdefault(str(key), str(value))
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        os.environ.setdefault(key.strip(), value.strip())

from smartblog_worker import ensure_checkpoints
ensure_checkpoints()
"

# ── Install systemd service ──
if [ "${1:-}" != "--no-service" ]; then
    echo ""
    echo "Installing systemd service (auto-start on boot)..."
    SERVICE_FILE="/etc/systemd/system/liveavatar-watchdog.service"
    cat > /tmp/liveavatar-watchdog.service << EOF
[Unit]
Description=SmartBlog LiveAvatar Watchdog
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$REPO_ROOT
Environment=HOME=$HOME
ExecStart=/bin/bash $REPO_ROOT/watchdog.sh
Restart=always
RestartSec=10
User=$(whoami)

[Install]
WantedBy=multi-user.target
EOF
    sudo cp /tmp/liveavatar-watchdog.service "$SERVICE_FILE"
    # Disable old service if exists
    sudo systemctl disable liveavatar-worker 2>/dev/null || true
    sudo systemctl stop liveavatar-worker 2>/dev/null || true
    sudo systemctl daemon-reload
    sudo systemctl enable liveavatar-watchdog
    sudo systemctl start liveavatar-watchdog
    echo "Watchdog service installed and started. Auto-starts on boot."
    echo "Logs: journalctl -u liveavatar-watchdog -f"
    echo "Worker log: tail -f $REPO_ROOT/watchdog.log"
else
    echo ""
    echo "Skipped service install (--no-service flag)"
    echo "Start manually: $VENV_DIR/bin/python $REPO_ROOT/smartblog_worker.py"
fi

echo ""
echo "=== Setup complete ==="
