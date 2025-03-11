#!/bin/bash

# Define environment name
ENV_NAME="env_emd"

echo "Creating Conda environment: $ENV_NAME"
# Initialize
# 6. Initialize Conda for bash
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Create the Conda environment with Python 3.10
conda create -y -n $ENV_NAME python=3.10

# Activate the environment
conda activate $ENV_NAME


echo "Installing pip packages from requirements.txt..."
pip install -r requirement.txt

echo "All packages are installed successfully!"