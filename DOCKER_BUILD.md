# Docker Build Instructions for ESMFold HuggingFace

## Base Image Recommendation

**Selected: `nvidia/cuda:12.1.0-runtime-ubuntu22.04`**

### Why This Base Image?

1. **Runtime vs Devel**:
   - Runtime image (~2GB base) vs Devel (~8GB base)
   - No compilation needed (no OpenFold!)
   - Contains only CUDA runtime libraries

2. **CUDA 12.1**:
   - Modern CUDA version
   - Compatible with PyTorch 2.x
   - Widely supported by current GPUs

3. **Ubuntu 22.04 LTS**:
   - Long-term support until 2027
   - Modern Python 3.10 available
   - Up-to-date system packages

4. **Multi-architecture Support**:
   - AMD64/x86_64 for cloud/datacenter
   - ARM64 support available if needed

### Size Comparison

| Image | Base Size | Final Size (est.) |
|-------|-----------|------------------|
| Original ESMFold (cuda:11.3-devel) | ~8GB | ~15GB |
| HF Version (cuda:12.1-runtime) | ~2GB | ~8GB |

## Building the Container

### Quick Build (Recommended)

```bash
# GPU-enabled version
docker build --platform linux/amd64 -f Dockerfile.hf -t esmfold-hf:latest .

# CPU-only version (smaller, slower)
docker build --platform linux/amd64 \
  --build-arg INSTALL_GPU=false \
  -f Dockerfile.hf \
  -t esmfold-hf:cpu .
```

### Using Docker Compose

```bash
# Build GPU version
docker-compose build esmfold-hf-gpu

# Build CPU version
docker-compose build esmfold-hf-cpu

# Build both
docker-compose build
```

### Build Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `PYTHON_VERSION` | `3.10` | Python version to install |
| `INSTALL_GPU` | `true` | Install GPU-enabled PyTorch |
| `TORCH_VERSION` | `2.1.0` | PyTorch version |
| `TORCH_CUDA` | `cu121` | CUDA version for PyTorch |

### Custom Build Examples

```bash
# Different Python version
docker build --platform linux/amd64 \
  --build-arg PYTHON_VERSION=3.11 \
  -f Dockerfile.hf \
  -t esmfold-hf:py311 .

# Different CUDA version
docker build --platform linux/amd64 \
  --build-arg TORCH_CUDA=cu118 \
  -f Dockerfile.hf \
  -t esmfold-hf:cu118 .
```

## Running the Container

### Basic Usage

```bash
# Test installation
docker run --gpus all esmfold-hf:latest

# Interactive shell
docker run --gpus all -it esmfold-hf:latest /bin/bash

# Run on your data
docker run --gpus all \
  -v $(pwd)/data/input:/data/input \
  -v $(pwd)/data/output:/data/output \
  esmfold-hf:latest \
  esm-fold-hf -i /data/input/sequences.fasta -o /data/output
```

### With Docker Compose

```bash
# Run test
docker-compose run esmfold-hf-gpu

# Run with custom command
docker-compose run esmfold-hf-gpu \
  esm-fold-hf -i /data/input/sequences.fasta -o /data/output --fp16
```

### Volume Mounts

The container uses three important volumes:

1. **`/data/input`**: Input FASTA files
2. **`/data/output`**: Output PDB files
3. **`/root/.cache/huggingface`**: Model cache (named volume in compose)

## Optimization Tips

### 1. Pre-download Models (Bake into Image)

Uncomment this line in Dockerfile.hf (line 68):

```dockerfile
RUN python -c "from transformers import EsmForProteinFolding; EsmForProteinFolding.from_pretrained('facebook/esmfold_v1')"
```

**Pros**: No download needed at runtime (~10GB model included)
**Cons**: Larger image size, longer build time

### 2. Use Multi-stage Build (Advanced)

```dockerfile
# Builder stage
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 as builder
# ... install everything ...

# Runtime stage
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
# ... copy only what's needed ...
```

### 3. Layer Caching

Order matters! The Dockerfile is optimized for caching:
1. System packages (rarely change)
2. PyTorch installation (large, rarely change)
3. Requirements (change occasionally)
4. Source code (changes frequently)

## Troubleshooting

### Build Failures

**Out of Disk Space**:
```bash
# Clean Docker cache
docker system prune -a

# Check disk usage
docker system df
```

**Network Timeouts**:
```bash
# Increase timeout
docker build --network-timeout 600 -f Dockerfile.hf -t esmfold-hf:latest .
```

**Platform Issues (Mac M1/M2)**:
```bash
# Force AMD64 emulation
docker build --platform linux/amd64 -f Dockerfile.hf -t esmfold-hf:latest .
```

### Runtime Issues

**GPU Not Detected**:
```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-runtime-ubuntu22.04 nvidia-smi

# If that fails, install nvidia-docker2
sudo apt-get install nvidia-docker2
sudo systemctl restart docker
```

**Out of Memory**:
```bash
# Use memory optimization flags
docker run --gpus all \
  -v $(pwd)/data:/data \
  esmfold-hf:latest \
  esm-fold-hf -i /data/input.fasta -o /data/output \
  --fp16 --chunk-size 32 --max-tokens-per-batch 512
```

## Container Registry

### Tagging for Registry

```bash
# Tag for Docker Hub
docker tag esmfold-hf:latest username/esmfold-hf:latest
docker tag esmfold-hf:latest username/esmfold-hf:v2.0.1

# Push to registry
docker push username/esmfold-hf:latest
docker push username/esmfold-hf:v2.0.1
```

### Using from Registry

```bash
# Pull and run
docker pull username/esmfold-hf:latest
docker run --gpus all username/esmfold-hf:latest
```

## Performance Benchmarks

Approximate prediction times on various hardware:

| Hardware | Sequence Length | Time | Settings |
|----------|----------------|------|----------|
| A100 40GB | 100 AA | 2s | Default |
| A100 40GB | 500 AA | 15s | Default |
| V100 32GB | 100 AA | 3s | Default |
| V100 32GB | 500 AA | 25s | --fp16 |
| RTX 3090 | 100 AA | 4s | --fp16 |
| CPU only | 100 AA | 5min | --cpu-only |

## Best Practices

1. **Use named volumes** for model caching to avoid re-downloading
2. **Mount input/output** directories rather than copying files
3. **Enable fp16** on Ampere+ GPUs for 2x speedup
4. **Use chunk-size** parameter for memory-constrained GPUs
5. **Batch sequences** by length for efficient processing

## Next Steps

- [ ] Push to container registry
- [ ] Set up CI/CD for automated builds
- [ ] Create Kubernetes deployment manifests
- [ ] Add GPU memory profiling
- [ ] Implement model caching optimization
