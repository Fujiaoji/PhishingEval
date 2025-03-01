#!/bin/bash

# Define environment name
ENV_NAME="env_visualphishnet"

echo "Creating Conda environment: $ENV_NAME"
# Initialize
# 6. Initialize Conda for bash
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Create the Conda environment with Python 3.10
conda create -y -n $ENV_NAME python=3.10

# Activate the environment
conda activate $ENV_NAME

# # Upgrade pip
# echo "Upgrading pip..."
# pip install --upgrade pip
# tensorflow
echo "Installing tensorflow..."
pip install tensorflow==2.10.0

echo "Installing pip packages from requirements.txt..."
pip install -r requirement.txt

echo "All packages are installed successfully!"