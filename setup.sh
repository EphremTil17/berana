#!/bin/bash
# -----------------------------------------------------------------------------
# Berana-Trans: Professional Environment Orchestrator
# -----------------------------------------------------------------------------
# This script handles automated, isolated environment creation with a focus
# on CUDA-accelerated LLM and Vision stacks.
#
# CUDA 13.0 SUPPORT (February 2026):
#   - PyTorch must be installed from the cu130 index, NOT PyPI default.
#   - Surya 0.17.x requires transformers <5.0.0 (5.x breaks SuryaDecoderConfig).
#   - System prerequisite: sudo apt install -y poppler-utils
# -----------------------------------------------------------------------------
set -e

# Dynamically calculate project paths
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_PATH="$PROJECT_ROOT/.venv"
VENV_PIP="$VENV_PATH/bin/pip"
VENV_PYTHON="$VENV_PATH/bin/python"

# --- DIRECTORY INITIALIZATION ---
# Pre-create canonical output stage roots to ensure they are owned by the current host user,
# preventing Docker (running as root) from creating them with restricted permissions.
mkdir -p \
  "$PROJECT_ROOT/output/layout_prep" \
  "$PROJECT_ROOT/output/layout_inference" \
  "$PROJECT_ROOT/output/layout_diagnostics" \
  "$PROJECT_ROOT/output/column_crops" \
  "$PROJECT_ROOT/output/ocr_runs/inference" \
  "$PROJECT_ROOT/output/ocr_runs/training" \
  "$PROJECT_ROOT/output/.registry"

# --- TERMINAL COLORS ---
BLUE='\033[0;34m'
SUCCESS='\033[0;32m'
WARNING='\033[0;33m'
ERROR='\033[0;31m'
RESET='\033[0m'

have_command() {
    command -v "$1" >/dev/null 2>&1
}

can_use_sudo() {
    if [ "$(id -u)" -eq 0 ]; then
        return 0
    fi
    if have_command sudo; then
        sudo -n true >/dev/null 2>&1
        return $?
    fi
    return 1
}

run_as_root() {
    if [ "$(id -u)" -eq 0 ]; then
        "$@"
    else
        sudo "$@"
    fi
}

ensure_apt_packages() {
    if ! have_command apt-get; then
        echo -e "${ERROR}Required system packages are missing and apt-get is unavailable on this host.${RESET}"
        echo -e "${ERROR}Install these packages manually before rerunning setup:${RESET}"
        echo -e "${ERROR}  build-essential g++ gcc cmake pkg-config python3-dev poppler-utils${RESET}"
        exit 1
    fi
    echo -e "${BLUE}Installing required system packages for OCR training...${RESET}"
    if [ "$(id -u)" -eq 0 ]; then
        apt-get update && apt-get install -y build-essential g++ gcc cmake pkg-config python3-dev poppler-utils
        return
    fi
    if have_command sudo; then
        if sudo apt-get update && sudo apt-get install -y build-essential g++ gcc cmake pkg-config python3-dev poppler-utils; then
            return
        fi
    fi
    echo -e "${ERROR}Failed to install required system packages automatically.${RESET}"
    echo -e "${ERROR}Rerun setup.sh with administrative privileges, or run this manually first:${RESET}"
    echo -e "${ERROR}  sudo apt-get update && sudo apt-get install -y build-essential g++ gcc cmake pkg-config python3-dev poppler-utils${RESET}"
    exit 1
}

echo -e "${BLUE}============================================================${RESET}"
echo -e "${BLUE} Berana-Trans: Liturgical Ge'ez Pipeline Setup ${RESET}"
echo -e "${BLUE}============================================================${RESET}"
echo -e "Project Root: $PROJECT_ROOT"

# 0. System Dependencies Check
echo -e "${BLUE}[0/5] Verifying system dependencies...${RESET}"
if ! have_command pdfinfo || ! have_command g++ || ! have_command gcc || ! have_command cmake || ! have_command pkg-config; then
    ensure_apt_packages
fi
if ! have_command pdfinfo; then
    echo -e "${ERROR}pdfinfo is still unavailable after package installation.${RESET}"
    exit 1
fi
if ! have_command g++; then
    echo -e "${ERROR}g++ is still unavailable after package installation.${RESET}"
    exit 1
fi
if ! have_command nvcc; then
    echo -e "${WARNING}Warning: CUDA Compiler (nvcc) not found.${RESET}"
    echo -e "${WARNING}CUDA toolkit installation is handled manually on this project.${RESET}"
    echo -e "${WARNING}OCR training can still proceed if your runtime/driver stack is already working.${RESET}"
    echo -e "${WARNING}If you need local GPU llama-cpp builds, install CUDA toolkit yourself first:${RESET}"
    echo -e "${WARNING}  sudo apt-get install -y cuda-toolkit-13-0${RESET}"
fi

# 1. Environment Creation
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${BLUE}[1/5] Establishing isolated virtual environment...${RESET}"
    python3 -m venv .venv
fi

# 2. Preparation
echo -e "${BLUE}[2/5] Synchronizing package managers...${RESET}"
"$VENV_PIP" install --upgrade pip setuptools wheel

# 3. PyTorch Installation (CUDA 13.0)
# IMPORTANT: PyTorch MUST be installed from the cu130 index BEFORE requirements.txt.
# The default PyPI torch/torchvision wheels do not include CUDA 13.0 support correctly.
echo -e "${BLUE}[3/5] Installing PyTorch Stack with CUDA 13.0 support...${RESET}"
"$VENV_PIP" install torch==2.10.0+cu130 torchvision==0.25.0+cu130 torchaudio==2.10.0+cu130 --index-url https://download.pytorch.org/whl/cu130

# 4. Dependency Manifest Execution
echo -e "${BLUE}[4/5] Executing researched dependency manifest...${RESET}"
# We install standard packages after torch to avoid PyPI overwriting our CUDA build.
"$VENV_PIP" install -r requirements.txt

# 5. Accelerator Compilation (llama-cpp-python)
# We target CUDA 13.x for Ampere (RTX 3060 Ti) and newer architectures.
# Cache compiled wheels so repeated setup runs don't rebuild from source every time.
LLAMA_VERSION="0.3.16"
PY_TAG="$("$VENV_PYTHON" -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
WHEELHOUSE="$PROJECT_ROOT/.cache/wheels/llama-cpp-python/${LLAMA_VERSION}"
mkdir -p "$WHEELHOUSE"

if ! have_command nvcc; then
    echo -e "${WARNING}[5/5] Skipping optional llama-cpp-python CUDA build because nvcc is unavailable.${RESET}"
    echo -e "${WARNING}This does not block OCR training. Install CUDA toolkit later if you need local GPU llama-cpp inference.${RESET}"
else
    echo -e "${SUCCESS}[5/5] Building/using cached llama-cpp-python wheel with CUDA...${RESET}"
    CUDA_WHEEL="$(find "$WHEELHOUSE" -maxdepth 1 -type f -name "llama_cpp_python-${LLAMA_VERSION}-${PY_TAG}-*-linux_x86_64.whl" | head -n 1 || true)"
    if [ -n "$CUDA_WHEEL" ]; then
        echo -e "${BLUE}Using cached CUDA wheel: $(basename "$CUDA_WHEEL")${RESET}"
        "$VENV_PIP" install "$CUDA_WHEEL"
    else
        echo -e "${BLUE}No cached CUDA wheel found. Building and caching...${RESET}"
        CMAKE_ARGS="-DGGML_CUDA=on" "$VENV_PIP" wheel "llama-cpp-python==${LLAMA_VERSION}" -w "$WHEELHOUSE"
        CUDA_WHEEL="$(find "$WHEELHOUSE" -maxdepth 1 -type f -name "llama_cpp_python-${LLAMA_VERSION}-${PY_TAG}-*-linux_x86_64.whl" | head -n 1)"
        "$VENV_PIP" install "$CUDA_WHEEL"
    fi
fi

# 6. Pre-commit Installation
echo -e "${BLUE}[6/6] Initializing Git hooks (pre-commit)...${RESET}"
if [ -d ".git" ]; then
    "$VENV_PATH/bin/pre-commit" install
else
    echo -e "${WARNING}Warning: .git directory not found. Skipping pre-commit hook installation.${RESET}"
fi

# 7. Verify CUDA Stack
echo -e "${BLUE}Verifying CUDA stack...${RESET}"
"$VENV_PYTHON" -c "
import torch
if torch.cuda.is_available():
    print(f'  ✅ PyTorch {torch.__version__} | CUDA {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)}')
else:
    print('  ⚠️  CUDA not available. Models will run on CPU (very slow).')
"

echo -e "${BLUE}============================================================${RESET}"
echo -e "${SUCCESS} Setup Completed Successfully.${RESET}"
echo -e " Environment Location: $VENV_PATH"
echo -e " To begin development: ${SUCCESS}source .venv/bin/activate${RESET}"
echo -e ""
echo -e " > For Label Studio Training:  ${SUCCESS}./tools/label_studio/setup_label_studio.sh${RESET}"
echo -e " > GPU acceleration tip:       ${SUCCESS}export TORCH_DEVICE=cuda${RESET}"
echo -e "${BLUE}============================================================${RESET}"
