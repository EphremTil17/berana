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

echo -e "${BLUE}============================================================${RESET}"
echo -e "${BLUE} Berana-Trans: Liturgical Ge'ez Pipeline Setup ${RESET}"
echo -e "${BLUE}============================================================${RESET}"
echo -e "Project Root: $PROJECT_ROOT"

# 0. System Dependencies Check
echo -e "${BLUE}[0/5] Verifying system dependencies...${RESET}"
if ! command -v pdfinfo &> /dev/null; then
    echo -e "${WARNING}Warning: poppler-utils not found. pdf2image requires it.${RESET}"
    echo -e "${WARNING}Install with: sudo apt install -y poppler-utils${RESET}"
fi

if ! command -v nvcc &> /dev/null; then
    echo -e "${WARNING}Warning: CUDA Compiler (nvcc) not found.${RESET}"
    echo -e "${WARNING}For GPU acceleration, install CUDA 13.0 Toolkit:${RESET}"
    echo -e "${WARNING}  sudo apt install -y cuda-toolkit-13-0${RESET}"
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

if ! command -v nvcc &> /dev/null; then
    echo -e "${WARNING}Warning: CUDA Compiler (nvcc) not found. Falling back to CPU-only.${RESET}"
    CPU_WHEEL="$(find "$WHEELHOUSE" -maxdepth 1 -type f -name "llama_cpp_python-${LLAMA_VERSION}-${PY_TAG}-*.whl" | grep -v 'linux_x86_64_cuda' | head -n 1 || true)"
    if [ -n "$CPU_WHEEL" ]; then
        echo -e "${BLUE}Using cached CPU wheel: $(basename "$CPU_WHEEL")${RESET}"
        "$VENV_PIP" install "$CPU_WHEEL"
    else
        echo -e "${BLUE}No cached CPU wheel found. Building and caching...${RESET}"
        "$VENV_PIP" wheel "llama-cpp-python==${LLAMA_VERSION}" -w "$WHEELHOUSE"
        CPU_WHEEL="$(find "$WHEELHOUSE" -maxdepth 1 -type f -name "llama_cpp_python-${LLAMA_VERSION}-${PY_TAG}-*.whl" | grep -v 'linux_x86_64_cuda' | head -n 1)"
        "$VENV_PIP" install "$CPU_WHEEL"
    fi
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
