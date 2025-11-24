#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Setup script for ESM with Hugging Face Transformers support

from setuptools import setup, find_packages

with open("esm/version.py") as infile:
    exec(infile.read())

with open("README.md") as f:
    readme = f.read()

# Core dependencies for the base package
install_requires = [
    "torch>=1.12.0",
]

# Optional dependency groups
extras = {
    # Original ESMFold with OpenFold
    "esmfold": [
        "biopython",
        "deepspeed==0.5.9",
        "dm-tree",
        "pytorch-lightning",
        "omegaconf",
        "ml-collections",
        "einops",
        "scipy",
    ],
    # Hugging Face Transformers-based ESMFold (modern approach)
    "esmfold_hf": [
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "biopython",
    ],
    # Development tools
    "dev": [
        "pytest",
        "black",
        "flake8",
        "ipython",
    ],
}

# Convenience extras
extras["all"] = list(set(sum(extras.values(), [])))
extras["full"] = extras["esmfold"] + extras["esmfold_hf"]

# Package structure
packages = [
    "esm",
    "esm.model",
    "esm.inverse_folding",
    "esm.esmfold",
    "esm.esmfold.v1",
    "esm.scripts"
]

package_dir = {
    "esm": "esm",
    "esm.model": "esm/model",
    "esm.inverse_folding": "esm/inverse_folding",
    "esm.esmfold.v1": "esm/esmfold/v1",
    "esm.scripts": "scripts"
}

setup(
    name="fair-esm",
    version=version,
    description="Evolutionary Scale Modeling (esm): Pretrained language models for proteins. From Facebook AI Research.",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Facebook AI Research",
    url="https://github.com/facebookresearch/esm",
    license="MIT",
    packages=packages,
    package_dir=package_dir,
    install_requires=install_requires,
    extras_require=extras,
    data_files=[("source_docs/esm", ["LICENSE", "README.md", "CODE_OF_CONDUCT.rst"])],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            # Original scripts
            "esm-extract=esm.scripts.extract:main",
            "esm-fold=esm.scripts.fold:main",
            # Hugging Face version
            "esm-fold-hf=esm.scripts.hf_fold:main",
        ]
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
