#!/bin/bash
# -----------------------------------------------------------------------------
# Berana-Trans: Professional Environment Orchestrator
# -----------------------------------------------------------------------------
# This script handles automated, isolated environment creation with a focus
# on CUDA-accelerated LLM and Vision stacks.
# -----------------------------------------------------------------------------
set -e

# Dynamically calculate project paths
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_PATH="$PROJECT_ROOT/.venv"
VENV_PIP="$VENV_PATH/bin/pip"
VENV_PYTHON="$VENV_PATH/bin/python"

# --- TERMINAL COLORS ---
BLUE='\033[0;34m'
SUCCESS='\033[0;32m'
WARNING='\033[0;33m'
RESET='\033[0m'

echo -e "${BLUE}============================================================${RESET}"
echo -e "${BLUE} Berana-Trans: Liturgical Ge'ez Pipeline Setup ${RESET}"
echo -e "${BLUE}============================================================${RESET}"
echo -e "Project Root: $PROJECT_ROOT"

# 1. Environment Creation
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${BLUE}[1/4] Establishing isolated virtual environment...${RESET}"
    python3 -m venv .venv
fi

# 2. Preparation
echo -e "${BLUE}[2/4] Synchronizing package managers...${RESET}"
"$VENV_PIP" install --upgrade pip setuptools wheel

# 3. Dependency Manifest Execution
echo -e "${BLUE}[3/4] Executing researched dependency manifest...${RESET}"
# We install standard packages directly. The manifest handles the strict Surya pins.
"$VENV_PIP" install -r requirements.txt

# 4. Accelerator Compilation (llama-cpp-python)
# We prioritize CUDA 12.x for Ampere (RTX 3060 Ti) stability.
if ! command -v nvcc &> /dev/null; then
    echo -e "${WARNING}Warning: CUDA Compiler (nvcc) not found. Falling back to CPU-only.${RESET}"
    "$VENV_PIP" install llama-cpp-python==0.3.16
else
    echo -e "${SUCCESS}[4/4] Compiling Accelerator (llama-cpp-python) with CUDA...${RESET}"
    # Verification of GPU architecture targeting is handled internally by CMake
    CMAKE_ARGS="-DGGML_CUDA=on" "$VENV_PIP" install llama-cpp-python==0.3.16 --force-reinstall --no-cache-dir
fi

# 5. Pre-commit Installation
echo -e "${BLUE}[5/5] Initializing Git hooks (pre-commit)...${RESET}"
if [ -d ".git" ]; then
    "$VENV_PATH/bin/pre-commit" install
else
    echo -e "${WARNING}Warning: .git directory not found. Skipping pre-commit hook installation.${RESET}"
fi

echo -e "${BLUE}============================================================${RESET}"
echo -e "${SUCCESS} Setup Completed Successfully.${RESET}"
echo -e " Environment Location: $VENV_PATH"
echo -e " To begin development: ${SUCCESS}source .venv/bin/activate${RESET}"
echo -e "${BLUE}============================================================${RESET}"
