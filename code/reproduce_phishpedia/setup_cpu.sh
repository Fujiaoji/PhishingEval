#!/bin/bash

# Define environment name
ENV_NAME="env_phishpedia"

echo "Creating Conda environment: $ENV_NAME"
# Initialize
# 6. Initialize Conda for bash
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Create the Conda environment with Python 3.8
conda create -y -n $ENV_NAME python=3.8

# Activate the environment
conda activate $ENV_NAME

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
# CUDA 11.1
# echo "Installing torch..."
# pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
# echo "Installing Detectron2..."
# pip install detectron2 -f "https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html"

echo "Installing torch and Detectron2..."
pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install detectron2 -f "https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.9/index.html"


echo "Installing pip packages from requirements.txt..."
pip install -r requirement.txt

echo "All packages are installed successfully!"