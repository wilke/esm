# GPU Testing Guide for ESMFold HuggingFace

This document provides instructions for testing the ESMFold HuggingFace container on a GPU-enabled remote machine.

## Prerequisites

### On Remote Machine

1. **NVIDIA GPU** with compute capability 6.0+ (Pascal or newer)
2. **NVIDIA Driver** installed (version 525+)
3. **NVIDIA Container Toolkit** installed
4. **Docker** installed and configured
5. **Sufficient GPU memory**: Recommended 8GB+ VRAM

### Verify GPU Setup

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## Quick Start

### 1. Pull the Container

```bash
docker pull dxkb/esmfold:hf
```

### 2. Run Installation Test

```bash
# Test with GPU
docker run --gpus all dxkb/esmfold:hf

# Expected output should show:
# - CUDA available: True
# - GPU name and memory
# - Installation test PASSED!
```

### 3. Prepare Test Data

Create a test FASTA file:

```bash
mkdir -p test_data/input test_data/output

cat > test_data/input/test.fasta << 'EOF'
>test_protein_short
MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG

>test_protein_medium
MKFLKFSLLTAVLLSVVFAFSSCGDDDDTYPYDVPDYAGYPYDVPDYAGYPYDVPDYAGMKFLKF

>test_protein_long
MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGGMKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGGMKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG
EOF
```

## GPU Testing Scenarios

### Test 1: Basic GPU Inference

```bash
docker run --gpus all \
  -v $(pwd)/test_data:/data \
  dxkb/esmfold:hf \
  esm-fold-hf -i /data/input/test.fasta -o /data/output
```

**Expected behavior:**
- Should detect GPU automatically
- Use fp16 and TF32 optimizations
- Process all sequences
- Generate PDB files in `test_data/output/`

**Metrics to check:**
- Time per sequence (should be fast with GPU)
- pLDDT scores (should be >70 for good predictions)
- pTM scores (should be >0.5 for confident predictions)

### Test 2: Memory Optimization

For GPUs with limited memory:

```bash
docker run --gpus all \
  -v $(pwd)/test_data:/data \
  dxkb/esmfold:hf \
  esm-fold-hf -i /data/input/test.fasta -o /data/output \
  --fp16 \
  --chunk-size 64 \
  --max-tokens-per-batch 512
```

**Parameters:**
- `--fp16`: Use half precision (saves ~50% memory)
- `--chunk-size 64`: Reduces memory from O(L²) to O(L)
- `--max-tokens-per-batch 512`: Process shorter sequences per batch

### Test 3: Performance Benchmarking

```bash
# Benchmark with timing
time docker run --gpus all \
  -v $(pwd)/test_data:/data \
  dxkb/esmfold:hf \
  esm-fold-hf -i /data/input/test.fasta -o /data/output --fp16
```

**Expected performance (approximate):**
| GPU | Sequence Length | Time | Settings |
|-----|----------------|------|----------|
| A100 40GB | 100 AA | 2s | Default |
| A100 40GB | 500 AA | 15s | Default |
| V100 32GB | 100 AA | 3s | Default |
| V100 32GB | 500 AA | 25s | --fp16 |
| RTX 3090 | 100 AA | 4s | --fp16 |
| T4 | 100 AA | 8s | --fp16 --chunk-size 64 |

### Test 4: Interactive Shell for Debugging

```bash
docker run --gpus all -it \
  -v $(pwd)/test_data:/data \
  dxkb/esmfold:hf \
  /bin/bash
```

Inside the container:

```bash
# Check GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')"

# Run test
python /workspace/test_installation.py

# Manual inference
esm-fold-hf -i /data/input/test.fasta -o /data/output --fp16

# Or use Python script
python /workspace/inference.py "MKTVRQERLK" --output test.pdb
```

## Troubleshooting

### Issue: CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Enable fp16:**
   ```bash
   esm-fold-hf -i input.fasta -o output/ --fp16
   ```

2. **Reduce chunk size:**
   ```bash
   esm-fold-hf -i input.fasta -o output/ --fp16 --chunk-size 32
   ```

3. **Lower batch size:**
   ```bash
   esm-fold-hf -i input.fasta -o output/ --fp16 --chunk-size 32 --max-tokens-per-batch 256
   ```

4. **Process one sequence at a time:**
   Split your FASTA file and process individually

### Issue: GPU Not Detected

**Symptoms:**
```
CUDA available: False
```

**Solutions:**

1. **Check NVIDIA driver:**
   ```bash
   nvidia-smi
   ```

2. **Verify Docker GPU access:**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

3. **Install NVIDIA Container Toolkit:**
   ```bash
   # Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

### Issue: Slow Performance

**Check:**

1. **GPU is being used:**
   ```bash
   # In another terminal while running
   watch -n 1 nvidia-smi
   ```
   Should show GPU utilization and memory usage

2. **Using optimizations:**
   Ensure you're using `--fp16` and `--use-tf32` flags

3. **Model is cached:**
   First run downloads ~10GB model, subsequent runs are faster

### Issue: Permission Denied on Output

**Solution:**
```bash
chmod -R 777 test_data/output
# Or run container with user mapping
docker run --gpus all --user $(id -u):$(id -g) \
  -v $(pwd)/test_data:/data \
  dxkb/esmfold:hf \
  esm-fold-hf -i /data/input/test.fasta -o /data/output
```

## Performance Monitoring

### Real-time GPU Monitoring

```bash
# Terminal 1: Run inference
docker run --gpus all \
  -v $(pwd)/test_data:/data \
  dxkb/esmfold:hf \
  esm-fold-hf -i /data/input/test.fasta -o /data/output

# Terminal 2: Monitor GPU
watch -n 0.5 nvidia-smi
```

### Check Results

```bash
# List generated PDB files
ls -lh test_data/output/

# View prediction quality from logs
grep "pLDDT\|pTM" <logfile>

# Visualize PDB (if PyMOL installed)
pymol test_data/output/*.pdb
```

## Production Deployment

### Using Docker Compose

Create `docker-compose.gpu.yml`:

```yaml
version: '3.8'

services:
  esmfold:
    image: dxkb/esmfold:hf
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    volumes:
      - ./input:/data/input
      - ./output:/data/output
      - esmfold-cache:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  esmfold-cache:
```

Run:
```bash
docker-compose -f docker-compose.gpu.yml run esmfold \
  esm-fold-hf -i /data/input/sequences.fasta -o /data/output --fp16
```

### Batch Processing Script

```bash
#!/bin/bash
# process_batch.sh

INPUT_DIR="/path/to/fasta/files"
OUTPUT_DIR="/path/to/output"

for fasta in $INPUT_DIR/*.fasta; do
    basename=$(basename "$fasta" .fasta)
    echo "Processing $basename..."

    docker run --gpus all \
      -v $INPUT_DIR:/data/input \
      -v $OUTPUT_DIR:/data/output \
      dxkb/esmfold:hf \
      esm-fold-hf \
        -i /data/input/$(basename "$fasta") \
        -o /data/output/$basename \
        --fp16 \
        --chunk-size 64

    echo "Completed $basename"
done
```

## Validation Tests

### Test Suite

```bash
#!/bin/bash
# run_validation.sh

echo "=== ESMFold HuggingFace GPU Validation ==="

# Test 1: GPU detection
echo "Test 1: GPU Detection"
docker run --gpus all dxkb/esmfold:hf python -c "
import torch
assert torch.cuda.is_available(), 'GPU not available'
print(f'✓ GPU detected: {torch.cuda.get_device_name(0)}')
"

# Test 2: Installation test
echo "Test 2: Installation Test"
docker run --gpus all dxkb/esmfold:hf

# Test 3: Single sequence inference
echo "Test 3: Single Sequence Inference"
mkdir -p test_output
echo ">test\nMKTVRQERLK" > test_seq.fasta
docker run --gpus all \
  -v $(pwd):/data \
  dxkb/esmfold:hf \
  esm-fold-hf -i /data/test_seq.fasta -o /data/test_output --fp16

# Test 4: Verify output
echo "Test 4: Verify Output"
if [ -f "test_output/test.pdb" ]; then
    echo "✓ PDB file generated successfully"
    wc -l test_output/test.pdb
else
    echo "✗ PDB file not found"
    exit 1
fi

echo "=== All tests passed! ==="
```

## Reporting Issues

When reporting issues, include:

1. **GPU information:**
   ```bash
   nvidia-smi
   ```

2. **Docker version:**
   ```bash
   docker --version
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

3. **Container logs:**
   ```bash
   docker run --gpus all dxkb/esmfold:hf 2>&1 | tee container.log
   ```

4. **Sequence information:**
   - Number of sequences
   - Average length
   - Example FASTA (if possible)

5. **Command used:**
   The exact docker run command

## Next Steps

After successful testing:

1. **Integrate into pipeline** - Use the validated container in production
2. **Scale up** - Test with larger datasets
3. **Optimize** - Fine-tune parameters for your specific GPU
4. **Monitor** - Set up logging and monitoring for production use

## Additional Resources

- [ESMFold Paper](https://www.science.org/doi/10.1126/science.ade2574)
- [HuggingFace Model](https://huggingface.co/facebook/esmfold_v1)
- [NVIDIA Container Toolkit Docs](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [Docker GPU Support](https://docs.docker.com/config/containers/resource_constraints/#gpu)
