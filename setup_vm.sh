#!/bin/bash
set -e

echo "=== AnticiPose VM Setup ==="

# ------------------------------------------------------------------
# Step 1: Install Python 3.8
# ------------------------------------------------------------------
echo "[1/5] Installing Python 3.8..."
apt update && apt install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt update
apt install -y python3.8 python3.8-venv python3.8-dev

# ------------------------------------------------------------------
# Step 2: Create Python 3.8 venv
# ------------------------------------------------------------------
echo "[2/5] Creating Python 3.8 venv..."
python3.8 -m venv /workspace/AnticiPose/.venv
source /workspace/AnticiPose/.venv/bin/activate
pip install --upgrade pip setuptools wheel

# ------------------------------------------------------------------
# Step 3: Download and install IsaacGym
# ------------------------------------------------------------------
echo "[3/5] Downloading and installing IsaacGym..."
pip install gdown
gdown --id 1iB6BJDD-tw7vFiWIwMttBYp8C7zdLHsH -O /workspace/IsaacGym_Preview_4_Package.tar.gz
tar xzf /workspace/IsaacGym_Preview_4_Package.tar.gz -C /workspace/
pip install -e /workspace/isaacgym/python

# ------------------------------------------------------------------
# Step 4: Install FALCON
# ------------------------------------------------------------------
echo "[4/5] Installing FALCON..."
cd /workspace/AnticiPose/FALCON
pip install -e .
pip install -e isaac_utils

# ------------------------------------------------------------------
# Step 5: Verify
# ------------------------------------------------------------------
echo "[5/5] Verifying installation..."
python -c "import isaacgym; print('IsaacGym OK')"
python -c "import humanoidverse; print('HumanoidVerse OK')"

echo ""
echo "=== Setup complete ==="
echo "Activate with: source /workspace/AnticiPose/.venv/bin/activate"
echo "Run from:      cd /workspace/AnticiPose/FALCON"
