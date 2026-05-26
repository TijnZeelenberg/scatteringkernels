#!/bin/bash

#SBATCH --job-name=impactparam_sweep
#SBATCH --partition=tue.gpu.q         # Choose a partition that has GPUs
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=1                      # This is how to request a GPU
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

# Set bash options for better error handling
set -euo pipefail

# Set enivornment
module purge
module load Python/3.13.1-GCCcore-14.2.0
module load uv
source .venv/bin/activate

# Execute the script or command
python training/parametersweeps/impactparam.py
