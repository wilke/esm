# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ESM (Evolutionary Scale Modeling) is a research repository from Meta's Fundamental AI Research Protein Team containing Transformer-based protein language models. The repository includes:

- **ESM-2**: State-of-the-art protein language models (8M to 15B parameters)
- **ESMFold**: End-to-end structure prediction from protein sequences
- **ESM-IF1**: Inverse folding model for sequence design from structures
- **ESM-1v**: Variant effect prediction models
- **MSA Transformer**: Multiple sequence alignment models

## Development Commands

### Installation

```bash
# Basic installation
pip install fair-esm

# ESMFold with dependencies (requires python <= 3.9, nvcc for OpenFold)
pip install "fair-esm[esmfold]"
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'

# From source (development)
pip install -e .
```

### Testing

```bash
# Run all tests
pytest tests/

# Run specific test files
pytest tests/test_alphabet.py
pytest tests/test_load_all.py
pytest tests/test_inverse_folding.py

# Test model loading (requires cached models)
pytest tests/test_load_all.py -k "test_load_hub_fwd_model"
```

### Code Quality

```bash
# Format code with black (line-length=99)
black esm/ scripts/ tests/
```

### Common Operations

```bash
# Extract embeddings from FASTA file
esm-extract esm2_t33_650M_UR50D examples/data/some_proteins.fasta output_dir --repr_layers 33 --include mean per_tok

# Alternative using script directly
python scripts/extract.py esm2_t33_650M_UR50D examples/data/some_proteins.fasta output_dir --repr_layers 0 32 33 --include mean per_tok

# Structure prediction with ESMFold
esm-fold -i input.fasta -o output_pdb_dir

# Inverse folding - sample sequences for a structure
python examples/inverse_folding/sample_sequences.py examples/inverse_folding/data/5YH2.pdb --chain C --temperature 1 --num-samples 3 --outpath output.fasta

# Inverse folding - score sequences
python examples/inverse_folding/score_log_likelihoods.py examples/inverse_folding/data/5YH2.pdb sequences.fasta --chain C --outpath scores.csv
```

## Architecture Overview

### Core Model Components

**Model hierarchy:**
- `esm/model/esm2.py` - ESM2 model (current SOTA)
- `esm/model/esm1.py` - Original ESM-1/ESM-1b models (ProteinBertModel)
- `esm/model/msa_transformer.py` - MSA Transformer
- `esm/esmfold/v1/esmfold.py` - ESMFold structure prediction
- `esm/inverse_folding/gvp_transformer.py` - ESM-IF1 inverse folding

**Key modules:**
- `esm/modules.py` - Shared transformer components (TransformerLayer, ContactPredictionHead, RobertaLMHead)
- `esm/multihead_attention.py` - Attention mechanisms
- `esm/rotary_embedding.py` - Rotary position embeddings (used in ESM-2)
- `esm/axial_attention.py` - Axial attention for ESMFold

### Data & I/O

**Core data classes (esm/data.py):**
- `Alphabet` - Token vocabulary and conversion
- `BatchConverter` - Converts sequences to tokenized batches
- `FastaBatchedDataset` - Efficient FASTA file loading with automatic batching

**Data flow:**
1. FASTA → FastaBatchedDataset
2. Sequences → Alphabet.get_batch_converter() → tokenized tensors
3. Tokens → Model → embeddings/predictions
4. Output saved as .pt files (torch.load to read)

### Model Loading

**Pretrained models (esm/pretrained.py):**
- `load_model_and_alphabet(name)` - Main loading function
- `load_model_and_alphabet_hub(name)` - Download from torch hub
- `load_model_and_alphabet_local(path)` - Load from local .pt file
- Models auto-download to `~/.cache/torch/hub/checkpoints/`

**Loading via PyTorch Hub:**
```python
import torch
model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
```

**Loading via esm.pretrained:**
```python
import esm
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
```

### Key Model Features

**ESM-2 specifics:**
- Uses rotary embeddings (no learned positional embeddings)
- Token dropout during training (mask_ratio_train = 0.15 * 0.8)
- Layer norm uses ESM1bLayerNorm variant
- Contact prediction via attention head combinations

**ESMFold specifics:**
- Combines ESM-2 language model (frozen/finetuned) with structure module
- Uses axial attention for memory efficiency
- Supports CPU offloading via `--cpu-offload` flag
- Can predict multimers (chains separated by ':')
- Chunk size parameter controls memory vs speed tradeoff

**Inverse Folding (ESM-IF1):**
- GVP (Geometric Vector Perceptrons) encoder for structure
- Transformer decoder for sequence generation
- Handles multichain structures
- Can tolerate missing backbone coordinates (span masking)

## Project Structure

```
esm/
├── esm/                          # Main package
│   ├── __init__.py              # Exports Alphabet, BatchConverter, models, pretrained
│   ├── data.py                  # Data loading (Alphabet, FastaBatchedDataset)
│   ├── pretrained.py            # Model loading utilities
│   ├── modules.py               # Shared transformer components
│   ├── model/                   # Model implementations
│   │   ├── esm1.py             # ESM-1/1b (ProteinBertModel)
│   │   ├── esm2.py             # ESM-2 (current SOTA)
│   │   └── msa_transformer.py   # MSA Transformer
│   ├── esmfold/v1/             # ESMFold structure prediction
│   │   ├── esmfold.py
│   │   ├── trunk.py
│   │   └── pretrained.py
│   └── inverse_folding/         # ESM-IF1 inverse folding
│       ├── gvp_transformer.py
│       ├── features.py
│       └── util.py
├── scripts/
│   ├── extract.py               # CLI for embedding extraction (esm-extract)
│   └── fold.py                  # CLI for structure prediction (esm-fold)
├── examples/                    # Example usage and research code
│   ├── inverse_folding/        # Inverse folding examples
│   ├── variant-prediction/     # Variant effect prediction
│   ├── lm-design/              # Protein design (Lin et al. 2022)
│   └── protein-programming-language/  # High-level design language
├── tests/                       # Unit tests
└── hubconf.py                  # PyTorch Hub entry points
```

## Important Design Patterns

**Model forward pass returns dict:**
```python
output = model(tokens, repr_layers=[33], return_contacts=True)
# output is dict with keys: 'logits', 'representations', 'contacts', 'attentions'
```

**Representation layers:**
- Layer 0 = initial embeddings (before transformer)
- Layer 1-N = after each transformer layer
- Specify `repr_layers=[0, 32, 33]` to extract multiple layers
- Per-sequence embeddings: average token embeddings (excluding BOS/EOS)

**Batch conversion:**
```python
data = [("protein1", "MKTV..."), ("protein2", "KALT...")]
batch_converter = alphabet.get_batch_converter()
batch_labels, batch_strs, batch_tokens = batch_converter(data)
```

**GPU/CPU handling:**
- Models check `torch.cuda.is_available()` but don't auto-move to GPU
- ESMFold supports CPU offloading for large sequences via FairScale FSDP
- See `examples/esm2_infer_fairscale_fsdp_cpu_offloading.py`

## Common Development Tasks

**Adding a new model variant:**
1. Implement model class in `esm/model/` (inherit from nn.Module)
2. Add pretrained loader function in `esm/pretrained.py`
3. Add entry in `hubconf.py` for PyTorch Hub support
4. Add test case in `tests/test_load_all.py`

**Extracting embeddings programmatically:**
```python
import torch
import esm

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

data = [("seq1", "MKTV...")]
batch_labels, batch_strs, batch_tokens = batch_converter(data)

with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33])
embeddings = results["representations"][33]
```

**Working with contact predictions:**
```python
# Unsupervised contact prediction via attention
results = model(tokens, return_contacts=True)
contact_map = results["contacts"]  # shape: [batch, seq_len, seq_len]
```

## Testing Guidelines

- Most tests in `tests/` assume models are already cached in `~/.cache/torch/hub/checkpoints/`
- `test_load_all.py` tests all model variants - can be slow
- `test_notebooks.py` validates Jupyter notebooks execute correctly
- Use `pytest -k "pattern"` to run specific test subsets

## Dependencies

**Core:**
- PyTorch (required)
- fairscale (for FSDP CPU offloading)

**ESMFold extras:**
- biopython, deepspeed==0.5.9, dm-tree, pytorch-lightning, omegaconf, ml-collections, einops, scipy
- openfold (requires nvcc/CUDA compiler)
- dllogger

**Development:**
- pytest (testing)
- black (formatting, line-length=99)

## Notes

- The repository builds on the fairseq framework but is standalone
- Models use Facebook/Meta's checkpoint format (.pt files)
- ESM-1v has 5 ensemble members (esm1v_t33_650M_UR90S_1 through _5)
- MSA Transformer takes 3D input: [batch, num_alignments, seq_len]
- Avoid using BOS token embeddings for downstream tasks (not trained with supervision)
