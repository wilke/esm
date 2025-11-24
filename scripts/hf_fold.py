#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Modern ESMFold CLI using Hugging Face Transformers
# Based on: https://huggingface.co/facebook/esmfold_v1

import argparse
import logging
import sys
from pathlib import Path
from timeit import default_timer as timer
from typing import List, Tuple, Optional

import torch
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%y/%m/%d %H:%M:%S",
)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def read_fasta(fasta_path: Path) -> List[Tuple[str, str]]:
    """
    Read sequences from a FASTA file.

    Returns:
        List of (header, sequence) tuples
    """
    sequences = []
    current_header = None
    current_seq = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_header is not None:
                    sequences.append((current_header, "".join(current_seq)))
                current_header = line[1:].split()[0]  # Take first word after >
                current_seq = []
            elif line:
                current_seq.append(line)

        if current_header is not None:
            sequences.append((current_header, "".join(current_seq)))

    return sequences


def convert_outputs_to_pdb(outputs: dict) -> List[str]:
    """
    Convert model outputs to PDB format strings.

    Args:
        outputs: Dictionary of model outputs containing positions, atom masks, etc.

    Returns:
        List of PDB strings, one per sequence in the batch
    """
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]

    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))

    return pdbs


def create_batched_sequences(
    sequences: List[Tuple[str, str]],
    max_tokens_per_batch: int = 1024
) -> List[Tuple[List[str], List[str]]]:
    """
    Group sequences into batches based on total token count.

    Args:
        sequences: List of (header, sequence) tuples
        max_tokens_per_batch: Maximum total sequence length per batch

    Yields:
        Tuples of (headers, sequences) for each batch
    """
    batch_headers, batch_sequences, num_tokens = [], [], 0

    for header, seq in sequences:
        if (len(seq) + num_tokens > max_tokens_per_batch) and num_tokens > 0:
            yield batch_headers, batch_sequences
            batch_headers, batch_sequences, num_tokens = [], [], 0
        batch_headers.append(header)
        batch_sequences.append(seq)
        num_tokens += len(seq)

    if batch_headers:
        yield batch_headers, batch_sequences


def create_parser():
    parser = argparse.ArgumentParser(
        description="ESMFold structure prediction using Hugging Face Transformers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-i", "--fasta",
        help="Path to input FASTA file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-o", "--pdb",
        help="Path to output PDB directory",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-m", "--model-name",
        help="Hugging Face model name or local path",
        type=str,
        default="facebook/esmfold_v1",
    )
    parser.add_argument(
        "--max-tokens-per-batch",
        type=int,
        default=1024,
        help="Maximum number of tokens per batch. Lower this for GPU memory issues.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size for axial attention to reduce memory usage. "
             "Recommended values: 128, 64, 32. Lower = less memory, slower speed.",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Run on CPU only (slower but works without GPU)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use half precision (fp16) for the language model to save memory",
    )
    parser.add_argument(
        "--use-tf32",
        action="store_true",
        help="Enable TensorFloat32 for faster computation on Ampere GPUs",
    )
    parser.add_argument(
        "--low-cpu-mem",
        action="store_true",
        default=True,
        help="Use low CPU memory mode during model loading",
    )

    return parser


def run(args):
    """Main execution function."""

    # Validate inputs
    if not args.fasta.exists():
        raise FileNotFoundError(f"FASTA file not found: {args.fasta}")

    args.pdb.mkdir(exist_ok=True, parents=True)

    # Determine device
    if args.cpu_only:
        device = "cpu"
        logger.info("Using CPU")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("CUDA not available, falling back to CPU")

    # Load sequences
    logger.info(f"Reading sequences from {args.fasta}")
    all_sequences = sorted(read_fasta(args.fasta), key=lambda x: len(x[1]))
    logger.info(f"Loaded {len(all_sequences)} sequences")

    # Load model and tokenizer
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = EsmForProteinFolding.from_pretrained(
        args.model_name,
        low_cpu_mem_usage=args.low_cpu_mem
    )

    # Apply optimizations
    if args.fp16 and device == "cuda":
        logger.info("Converting language model to fp16")
        model.esm = model.esm.half()

    if args.use_tf32 and device == "cuda":
        logger.info("Enabling TensorFloat32")
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.chunk_size is not None:
        logger.info(f"Setting chunk size to {args.chunk_size}")
        model.trunk.set_chunk_size(args.chunk_size)

    model = model.to(device)
    model.eval()

    logger.info("Starting predictions")

    # Process batches
    batches = list(create_batched_sequences(all_sequences, args.max_tokens_per_batch))
    num_completed = 0
    num_sequences = len(all_sequences)

    for batch_idx, (headers, sequences) in enumerate(batches):
        start = timer()

        try:
            # Tokenize
            tokenized_input = tokenizer(
                sequences,
                return_tensors="pt",
                add_special_tokens=False
            )['input_ids']
            tokenized_input = tokenized_input.to(device)

            # Run inference
            with torch.no_grad():
                output = model(tokenized_input)

            # Convert to PDB
            pdbs = convert_outputs_to_pdb(output)

            # Calculate metrics
            mean_plddts = output["plddt"].mean(dim=1).cpu().numpy()
            ptms = output["ptm"].cpu().numpy() if "ptm" in output else [0.0] * len(sequences)

            # Save outputs
            tottime = timer() - start
            time_string = f"{tottime / len(headers):.1f}s"
            if len(sequences) > 1:
                time_string = time_string + f" (amortized, batch size {len(sequences)})"

            for header, seq, pdb_string, mean_plddt, ptm in zip(
                headers, sequences, pdbs, mean_plddts, ptms
            ):
                output_file = args.pdb / f"{header}.pdb"
                output_file.write_text(pdb_string)
                num_completed += 1

                logger.info(
                    f"Predicted structure for {header} with length {len(seq)}, "
                    f"pLDDT {mean_plddt:.1f}, pTM {ptm:.3f} in {time_string}. "
                    f"{num_completed} / {num_sequences} completed."
                )

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if len(sequences) > 1:
                    logger.error(
                        f"Out of memory error on batch {batch_idx + 1} of size {len(sequences)}. "
                        "Try lowering --max-tokens-per-batch or using --chunk-size."
                    )
                else:
                    logger.error(
                        f"Out of memory on sequence {headers[0]} of length {len(sequences[0])}. "
                        "Try using --chunk-size or --cpu-only."
                    )
                continue
            raise

        except Exception as e:
            logger.error(f"Error processing batch {batch_idx + 1}: {e}")
            raise

    logger.info(f"Completed! Predicted {num_completed} / {num_sequences} structures")


def main():
    parser = create_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
