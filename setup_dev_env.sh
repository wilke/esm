#!/bin/bash
# Development environment setup script for ESM with HuggingFace support
# Usage: ./setup_dev_env.sh [environment_name]

set -e  # Exit on error

ENV_NAME="${1:-esm-hf-dev}"
PYTHON_VERSION="3.10"

echo "=========================================="
echo "ESM HuggingFace Development Environment"
echo "=========================================="
echo ""
echo "Environment name: $ENV_NAME"
echo "Python version: $PYTHON_VERSION"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Miniconda or Anaconda."
    exit 1
fi

# Create conda environment
echo "Creating conda environment: $ENV_NAME"
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

echo ""
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Install PyTorch (CPU version - users can reinstall for CUDA)
echo ""
echo "Installing PyTorch (CPU version)..."
echo "Note: For GPU support, reinstall PyTorch from pytorch.org with your CUDA version"
conda install pytorch cpuonly -c pytorch -y

# Install the package in editable mode with HuggingFace extras
echo ""
echo "Installing ESM in editable mode with HuggingFace support..."
python -m pip install -e ".[esmfold_hf,dev]" -f setup_hf.py

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
python -c "import esm; print(f'ESM version: {esm.__version__}')" || echo "Warning: Could not import esm"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')" || echo "Warning: Could not import transformers"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" || echo "Warning: Could not import torch"

# Check if CLI tools are available
echo ""
if command -v esm-fold-hf &> /dev/null; then
    echo "✓ esm-fold-hf CLI installed successfully"
else
    echo "✗ esm-fold-hf CLI not found"
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To use ESMFold with HuggingFace:"
echo "  esm-fold-hf -i input.fasta -o output_dir/"
echo ""
echo "For GPU support, reinstall PyTorch:"
echo "  conda activate $ENV_NAME"
echo "  # Visit pytorch.org for the right command for your CUDA version"
echo "  # Example for CUDA 11.8:"
echo "  conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia"
echo ""
