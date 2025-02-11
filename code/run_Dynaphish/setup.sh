#!/bin/bash

# Source the Conda configuration
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Check if the environment already exists
conda info --envs | grep -w "$ENV_NAME" > /dev/null

if [ $? -eq 0 ]; then
    echo "Activating Conda environment $ENV_NAME"
    conda activate "$ENV_NAME"
else
    echo "Creating and activating new Conda environment $ENV_NAME with Python 3.7"
    conda create -n "$ENV_NAME" python=3.8
    conda activate "$ENV_NAME"
fi

## Install MyXDriver
PACKAGE_NAME="xdriver"
installed_packages=$(conda run -n "$ENV_NAME" conda list)
if echo "$installed_packages" | grep -q "$PACKAGE_NAME"; then
  echo "MyXdriver_pub is already installed, skip installation"
else
  git clone https://github.com/lindsey98/MyXdriver_pub.git
  cd MyXdriver_pub
#   chmod +x bash setup.sh
  export ENV_NAME="$ENV_NAME" && bash setup.sh
  cd ../
fi

echo "All packages installed successfully!"