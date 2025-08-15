#!/bin/bash

# Exit immediately on error
set -e

# Step 1: Create the conda environment
echo "Creating conda environment 'maven_analysis' with Python 3.10..."
conda create -n maven_analysis python=3.10 -y

# Step 2: Activate the environment
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate maven_analysis

# Step 3: Install Jupyter and ipykernel
echo "Installing Jupyter and ipykernel..."
pip install --upgrade pip
pip install jupyter ipykernel

# Step 4: Register the environment as a Jupyter kernel
echo "Registering Jupyter kernel..."
python -m ipykernel install --user --name=maven_analysis --display-name "MAVEN analysis environment"

# Step 5: Install from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "✅ Setup complete! You can now launch Jupyter and select 'MAVEN analysis environment' as the kernel."

