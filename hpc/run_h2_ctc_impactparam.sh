#!/bin/bash
#SBATCH --job-name=b_ctc
#SBATCH --partition=tue.cpu1.q        # verify with: sinfo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2gb
#SBATCH --time=04:00:00
#SBATCH --output=hpc/logs/%x%j.out
#SBATCH --error=hpc/logs/%x%j.err
#SBATCH --chdir=/home/20193567/scatteringkernels

# Set bash options for better error handling
set -euo pipefail

# Set enivornment
module purge
module load Python/3.13.1-GCCcore-14.2.0
module load uv
source .venv/bin/activate

# Pin all threading layers to the Slurm allocation so no layer steals extra
# cores from other jobs on the same node.
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Using $NUMBA_NUM_THREADS threads (SLURM_CPUS_PER_TASK)"
echo "Working directory: $(pwd)"

# Run as module to ensure correct working directory and environment on hpc
uv run python -m h2_ctc.ctc_impactparamsweep
