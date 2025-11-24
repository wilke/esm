# ESM with Hugging Face Transformers Support

This branch adds modern Hugging Face Transformers-based ESMFold support to the ESM repository. This provides a simpler, more maintainable alternative to the original OpenFold-based implementation.

## Quick Start

### Installation

#### Option 1: Quick Setup with Script (Recommended)

```bash
# Create and setup development environment
./setup_dev_env.sh esm-hf-dev

# Activate the environment
conda activate esm-hf-dev
```

#### Option 2: Manual Installation

```bash
# Create conda environment
conda create -n esm-hf python=3.10 -y
conda activate esm-hf

# Install PyTorch (visit pytorch.org for GPU-specific instructions)
conda install pytorch cpuonly -c pytorch -y

# Install ESM with HuggingFace support
pip install -e ".[esmfold_hf]" -f setup_hf.py

# Or just install requirements
pip install -r requirements_hf.txt
pip install -e .
```

### Usage

#### Command Line Interface

```bash
# Basic usage
esm-fold-hf -i input.fasta -o output_pdb_dir/

# With GPU memory optimization
esm-fold-hf -i input.fasta -o output_pdb_dir/ --fp16 --chunk-size 64

# CPU-only mode
esm-fold-hf -i input.fasta -o output_pdb_dir/ --cpu-only

# Advanced options
esm-fold-hf \
  -i input.fasta \
  -o output_pdb_dir/ \
  --max-tokens-per-batch 512 \
  --chunk-size 32 \
  --fp16 \
  --use-tf32
```

#### Python API

```python
from transformers import AutoTokenizer, EsmForProteinFolding
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")

# Optional: Enable optimizations
model.esm = model.esm.half()  # Use fp16
model.trunk.set_chunk_size(64)  # Reduce memory usage
model = model.cuda()
model.eval()

# Predict structure
sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
tokenized = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)['input_ids']
tokenized = tokenized.cuda()

with torch.no_grad():
    output = model(tokenized)

# Access predictions
positions = output["positions"]  # 3D coordinates
plddt = output["plddt"]  # Confidence scores per residue
mean_plddt = plddt.mean()  # Overall confidence

print(f"Predicted structure with mean pLDDT: {mean_plddt:.2f}")
```

## Features

### Advantages of HuggingFace Implementation

- **Simpler Installation**: No OpenFold compilation or CUDA dependencies
- **Better Memory Management**: Uses Hugging Face's optimized attention mechanisms
- **Modern Codebase**: Built on transformers library with active maintenance
- **Easy Model Switching**: Can easily use different model checkpoints
- **Better Documentation**: Leverages Hugging Face's extensive documentation

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --fasta` | Input FASTA file (required) | - |
| `-o, --pdb` | Output PDB directory (required) | - |
| `-m, --model-name` | HuggingFace model name or local path | `facebook/esmfold_v1` |
| `--max-tokens-per-batch` | Max tokens per batch (lower for less memory) | 1024 |
| `--chunk-size` | Axial attention chunk size (32, 64, 128) | None |
| `--cpu-only` | Force CPU execution | False |
| `--fp16` | Use half precision for language model | False |
| `--use-tf32` | Enable TensorFloat32 (Ampere GPUs) | False |
| `--low-cpu-mem` | Low CPU memory mode during loading | True |

### Memory Optimization Tips

1. **For GPU memory issues**:
   - Use `--fp16` to halve language model memory
   - Set `--chunk-size 64` or `32` to reduce attention memory
   - Lower `--max-tokens-per-batch` to process shorter sequences

2. **For very long sequences**:
   - Use `--chunk-size 32` (slower but much less memory)
   - Consider splitting into domains if biologically appropriate

3. **For CPU-only execution**:
   - Use `--cpu-only` flag
   - Be patient - CPU inference is much slower
   - The model automatically converts to fp32 for CPU

## Development

### Setup Development Environment

```bash
# Install with development dependencies
pip install -r requirements_dev.txt

# Or using setup_hf.py
pip install -e ".[esmfold_hf,dev]" -f setup_hf.py
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_load_all.py
```

### Code Formatting

```bash
# Format code with black (line-length=99)
black scripts/hf_fold.py
```

## Comparison: Original vs HuggingFace

| Feature | Original (`esm-fold`) | HuggingFace (`esm-fold-hf`) |
|---------|----------------------|----------------------------|
| Installation | Complex (OpenFold + CUDA) | Simple (pip install) |
| Dependencies | Heavy (deepspeed, etc.) | Lightweight (transformers) |
| Model Source | Fair-ESM checkpoint | HuggingFace Hub |
| Memory Usage | Higher | Optimized |
| Maintenance | Frozen codebase | Active HF updates |
| Documentation | Limited | Extensive (HF docs) |

## Requirements

### Minimum Requirements
- Python >= 3.8
- PyTorch >= 1.12.0
- transformers >= 4.30.0
- accelerate >= 0.20.0

### Recommended
- CUDA-capable GPU with 8GB+ VRAM
- 16GB+ system RAM
- SSD storage for model caching

## Troubleshooting

### Out of Memory Errors

```bash
# Try these in order:
esm-fold-hf -i input.fasta -o output/ --fp16
esm-fold-hf -i input.fasta -o output/ --fp16 --chunk-size 64
esm-fold-hf -i input.fasta -o output/ --fp16 --chunk-size 32 --max-tokens-per-batch 512
esm-fold-hf -i input.fasta -o output/ --cpu-only
```

### Model Download Issues

```bash
# Pre-download model
python -c "from transformers import EsmForProteinFolding; EsmForProteinFolding.from_pretrained('facebook/esmfold_v1')"

# Use local model
esm-fold-hf -i input.fasta -o output/ --model-name /path/to/local/model
```

### Import Errors

```bash
# Ensure all dependencies are installed
pip install transformers accelerate biopython torch

# Verify installation
python -c "import transformers; print(transformers.__version__)"
```

## Citation

If you use ESMFold in your research, please cite:

```bibtex
@article{lin2022language,
  title={Language models of protein sequences at the scale of evolution enable accurate structure prediction},
  author={Lin, Zeming and Akin, Halil and Rao, Roshan and Hie, Brian and Zhu, Zhongkai and Lu, Wenting and Smetanin, Nikita and Verkuil, Robert and Kabeli, Ori and Shmueli, Yilun and others},
  journal={Science},
  year={2022}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Additional Resources

- [Original ESM Repository](https://github.com/facebookresearch/esm)
- [HuggingFace ESMFold Model](https://huggingface.co/facebook/esmfold_v1)
- [HuggingFace Tutorial Notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/protein_folding.ipynb)
- [ESMFold Paper](https://www.science.org/doi/10.1126/science.ade2574)
