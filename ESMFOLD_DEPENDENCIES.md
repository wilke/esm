# ESMFold Dependencies

This document provides a comprehensive list of dependencies required to run ESMFold.

## System Requirements

- **Python**: <= 3.9 (3.7-3.9 recommended)
- **CUDA**: 11.3 or compatible (for GPU support)
- **NVCC**: CUDA compiler (required for building OpenFold)
- **Operating System**: Linux (recommended), macOS (limited support)

## Core Dependencies

### PyTorch and CUDA
- `torch>=1.12` (with CUDA 11.3+ support)
- `cudatoolkit==11.3.*` (for GPU operations)

### ESM Package
- `fair-esm==2.0.1` (or latest)

## ESMFold-Specific Dependencies

### Required Python Packages (via pip)

**Core scientific packages:**
- `numpy==1.21.2` (or compatible)
- `scipy==1.7.1`
- `biopython==1.79`

**Deep learning frameworks:**
- `pytorch-lightning==1.5.10`
- `deepspeed==0.5.9`
- `fairscale` (for FSDP CPU offloading)

**Utilities:**
- `einops` (tensor operations)
- `omegaconf` (configuration management)
- `ml-collections==0.1.0`
- `dm-tree==0.1.6` (DeepMind tree utilities)

**Monitoring and utilities:**
- `tqdm==4.62.2` (progress bars)
- `wandb==0.12.21` (experiment tracking, optional)
- `PyYAML==5.4.1`
- `typing-extensions==3.10.0.2`
- `requests==2.26.0`

### OpenFold Dependencies

**Critical for structure prediction:**
- `openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307`
  - Specific commit hash for compatibility
  - Requires NVCC (CUDA compiler) for installation
  - Provides structure module and featurization utilities

**NVIDIA Deep Learning Logger:**
- `dllogger @ git+https://github.com/NVIDIA/dllogger.git`

### Optional Bioinformatics Tools (for MSA generation)

These are only needed if generating MSAs (not required for basic ESMFold inference):
- `openmm==7.5.1` (molecular dynamics)
- `pdbfixer` (PDB file cleanup)
- `hmmer==3.3.2` (sequence search)
- `hhsuite==3.3.0` (HMM-HMM search)
- `kalign2==2.04` (multiple sequence alignment)

## Installation Order

1. **Install PyTorch with CUDA support:**
   ```bash
   pip install torch==1.12.1+cu113 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
   ```

2. **Install ESM with ESMFold extras:**
   ```bash
   pip install "fair-esm[esmfold]"
   ```

3. **Install OpenFold and dllogger:**
   ```bash
   pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
   pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'
   ```

## Minimal Dependencies (Inference Only)

For running ESMFold inference without training capabilities:

```
torch>=1.12 (with CUDA)
fair-esm[esmfold]
dllogger
openfold
biopython
scipy
einops
omegaconf
ml-collections
dm-tree
```

## Docker Considerations

When building a Docker image:
- Use CUDA base image (e.g., `nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04`)
- Install NVCC and build-essential for compiling OpenFold
- Install PyTorch with CUDA support before other packages
- Install git for fetching OpenFold and dllogger
- Set environment variables for CUDA paths

## Known Issues

1. **OpenFold Installation**: Requires NVCC. If installation fails, verify:
   - `nvcc --version` shows CUDA compiler
   - PyTorch CUDA version matches system CUDA version
   - Sufficient disk space for compilation

2. **Python Version**: Python 3.10+ may have compatibility issues with some dependencies

3. **Memory Requirements**: ESMFold models (especially 3B parameter version) require significant GPU memory:
   - 8GB+ VRAM for small sequences
   - 16GB+ VRAM for longer sequences
   - CPU offloading available for memory-constrained systems

## Version Compatibility Matrix

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.7-3.9 | 3.9 recommended |
| PyTorch | 1.12.x | With CUDA 11.3+ |
| CUDA | 11.3+ | For GPU support |
| fair-esm | 2.0.1+ | Latest stable |
| openfold | commit 4b41059 | Specific hash required |
| deepspeed | 0.5.9 | Exact version |
| pytorch-lightning | 1.5.10 | Compatibility tested |
