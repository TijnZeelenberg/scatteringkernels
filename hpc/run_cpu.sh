#!/bin/bash
#SBATCH --job-name=ctc_h2_data
#SBATCH --partition=tue.cpu.q        # verify with: sinfo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2gb
#SBATCH --time=04:00:00
#SBATCH --output=hpc/logs/data_gen_%j.out
#SBATCH --error=hpc/logs/data_gen_%j.err

# Set bash options for better error handling
set -euo pipefail

# Set enivornment
module purge
module load Python/3.13.1-GCCcore-14.2.0
source .venv/bin/activate

# Pin all threading layers to the Slurm allocation so no layer steals extra
# cores from other jobs on the same node.
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Using $NUMBA_NUM_THREADS threads (SLURM_CPUS_PER_TASK)"
echo "Working directory: $(pwd)"

python ctc_adjusted/ctc_h2_impactparamsweep.py
