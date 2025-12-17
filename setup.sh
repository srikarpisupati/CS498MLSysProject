#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
CONDA_INSTALL_DIR="$HOME/miniconda3"
CONDA_BIN="$CONDA_INSTALL_DIR/bin/conda"

echo "======================================================================"
echo "ML Compiler Benchmark Framework - Setup Script"
echo "======================================================================"
echo ""

if [ ! -f "$CONDA_BIN" ]; then
    echo "Conda not found. Installing Miniconda..."
    
    INSTALLER="$PROJECT_DIR/Miniconda3-latest-Linux-x86_64.sh"
    if [ ! -f "$INSTALLER" ]; then
        echo "Downloading Miniconda installer..."
        cd "$PROJECT_DIR"
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        INSTALLER="$PROJECT_DIR/Miniconda3-latest-Linux-x86_64.sh"
        chmod +x "$INSTALLER"
    fi
    
    echo "Installing Miniconda to $CONDA_INSTALL_DIR..."
    bash "$INSTALLER" -b -p "$CONDA_INSTALL_DIR"
    
    echo "Initializing conda..."
    "$CONDA_BIN" init bash
    
    echo "Miniconda installed successfully!"
else
    echo "Conda already installed at $CONDA_INSTALL_DIR"
fi

export PATH="$CONDA_INSTALL_DIR/bin:$PATH"

if [ -f "$CONDA_INSTALL_DIR/etc/profile.d/conda.sh" ]; then
    source "$CONDA_INSTALL_DIR/etc/profile.d/conda.sh"
fi

echo ""
echo "Accepting conda Terms of Service..."
"$CONDA_BIN" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
"$CONDA_BIN" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

cd "$PROJECT_DIR"

if [ ! -f "environment.yml" ]; then
    echo "Error: environment.yml not found"
    exit 1
fi

ENV_NAME=$(grep "^name:" environment.yml | cut -d' ' -f2)
if conda env list | grep -q "^$ENV_NAME "; then
    echo ""
    echo "Removing existing environment '$ENV_NAME' for clean reinstall..."
    conda env remove -n "$ENV_NAME" -y || true
fi

echo ""
echo "Creating conda environment from environment.yml..."
echo "This may take several minutes as it downloads large packages (PyTorch, CUDA, etc.)..."
"$CONDA_BIN" env create -f environment.yml

TVM_WHEEL="$PROJECT_DIR/tlcpack_cu116-0.11.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
if [ -f "$TVM_WHEEL" ]; then
    echo ""
    echo "Installing TVM (TLCPack) wheel..."
    "$CONDA_INSTALL_DIR/envs/$ENV_NAME/bin/pip" install "$TVM_WHEEL"
else
    echo ""
    echo "⚠ TVM wheel not found at $TVM_WHEEL"
    echo "  Download a CUDA-enabled TLCPack wheel and place it in the project root before running setup."
fi

if conda env list | grep -q "^$ENV_NAME "; then
    echo ""
    echo "Checking GPU availability..."
    "$CONDA_INSTALL_DIR/envs/$ENV_NAME/bin/python" "$PROJECT_DIR/check_gpu.py" 2>/dev/null || echo "  (Skipping GPU check - run check_gpu.py manually)"

    echo ""
    echo "Checking NVCC..."
    if "$CONDA_INSTALL_DIR/envs/$ENV_NAME/bin/nvcc" --version >/dev/null 2>&1; then
        "$CONDA_INSTALL_DIR/envs/$ENV_NAME/bin/nvcc" --version | head -n 1
    else
        echo "  ⚠ nvcc not found. Ensure cuda-toolkit installed correctly."
    fi
    
    echo ""
    echo "======================================================================"
    echo "✓ Setup complete!"
    echo "======================================================================"
    echo ""
    echo "To activate the environment, run:"
    echo "  conda activate $ENV_NAME"
    echo ""
    echo "Or to run the benchmark directly:"
    echo "  $CONDA_INSTALL_DIR/envs/$ENV_NAME/bin/python $PROJECT_DIR/run_benchmark.py"
    echo ""
    echo "To check GPU status:"
    echo "  $CONDA_INSTALL_DIR/envs/$ENV_NAME/bin/python $PROJECT_DIR/check_gpu.py"
    echo ""
    echo "To use conda in new terminal sessions, reload your shell or run:"
    echo "  source ~/.bashrc"
    echo ""
else
    echo "Error: Environment was not created"
    exit 1
fi

