#!/bin/bash

#SBATCH --job-name=impactparam
#SBATCH --partition=tue.gpu1.q         # Choose a partition that is free
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G                     # request enough RAM for the job
#SBATCH --gpus=1                      # This is how to request a GPU
#SBATCH --output=hpc/logs/%x%j.out
#SBATCH --error=hpc/logs/%x%j.err
#SBATCH --chdir=/home/20193567/scatteringkernels
#SBATCH --dependency=afterany:5564367

# Set bash options for better error handling
set -euo pipefail

# Set enivornment
module purge
module load Python/3.13.1-GCCcore-14.2.0
module load uv
source .venv/bin/activate

# Execute the script or command
uv run python -m training.parametersweeps.impactparam
