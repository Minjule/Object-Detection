#!/usr/bin/env bash
set -euo pipefail

# ----------- User-editable -----------
PYTHON_BIN=python3            # e.g. python3 or python3.10
PYTHON_VER=$($PYTHON_BIN -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
# location to clone ONNX Runtime
WORKDIR=$HOME/onnxruntime_build
# optional: limit parallel jobs if OOM
PARALLEL_JOBS=1
# ------------------------------------

echo "Build ONNX Runtime (TensorRT EP) on Jetson Nano"
echo "Workdir: $WORKDIR"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# 1) Install system deps (adapt to your JetPack / Ubuntu)
sudo apt update
sudo apt install -y --no-install-recommends \
    build-essential git cmake curl ca-certificates \
    libopenblas-dev pkg-config libpython3-dev python3-pip python3-setuptools python3-wheel \
    libssl-dev

# 2) (Optional but recommended) increase swap if you have 2GB Nano
# Creates a 4GB swap file (adjust size if needed)
SWAPFILE=/var/swapfile-onnx
if [ ! -f "$SWAPFILE" ]; then
  echo "Creating 4G swap (may be needed building on 2GB Nano)..."
  sudo fallocate -l 4G $SWAPFILE
  sudo chmod 600 $SWAPFILE
  sudo mkswap $SWAPFILE
  sudo swapon $SWAPFILE
fi

# 3) Ensure CUDA & TensorRT libs are discoverable (JetPack installs them under /usr)
export PATH="/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"

# 4) Clone ONNX Runtime (recursive)
if [ ! -d onnxruntime ]; then
  git clone --recursive https://github.com/microsoft/onnxruntime.git
fi
cd onnxruntime

# (Optional) checkout a stable tag known to work with your JetPack/TensorRT - skip if you want latest
# git fetch --all --tags
# git checkout tags/v1.15.1 -b v1.15.1

# 5) Install Python build-time deps
$PYTHON_BIN -m pip install --upgrade pip setuptools wheel
$PYTHON_BIN -m pip install numpy packaging

# 6) Build command (GPU + TensorRT). Tune flags if OOM.
# NOTE: --cudnn_home and --tensorrt_home usually point to /usr/lib/aarch64-linux-gnu on Jetson
./build.sh --config Release --update --build --parallel ${PARALLEL_JOBS} --build_wheel \
  --use_cuda --cuda_home /usr/local/cuda \
  --cudnn_home /usr/lib/aarch64-linux-gnu \
  --use_tensorrt --tensorrt_home /usr/lib/aarch64-linux-gnu \
  --skip_tests \
  --cmake_extra_defines "CMAKE_CUDA_ARCHITECTURES=native;onnxruntime_BUILD_UNIT_TESTS=OFF;onnxruntime_USE_FLASH_ATTENTION=OFF;onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION=OFF"

# If build fails with OOM/hangs, retry with --parallel 1 or add more swap, or pass --build_dir to build into a different path.

# 7) After successful build, find wheel and install
WHEEL=$(find build -type f -name "onnxruntime*-gpu*.whl" | head -n 1)
if [ -z "$WHEEL" ]; then
  echo "Wheel not found. Check build output under build/Linux/Release/dist/"
  exit 1
fi
echo "Installing wheel: $WHEEL"
$PYTHON_BIN -m pip install --upgrade "$WHEEL"

echo "DONE. ONNX Runtime GPU (TensorRT EP) installed."

# 8) (Optional) disable swap file if you created one (you may keep it)
# sudo swapoff $SWAPFILE && sudo rm -f $SWAPFILE
