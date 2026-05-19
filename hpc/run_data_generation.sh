#!/bin/bash
#SBATCH --job-name=ctc_h2_data
#SBATCH --partition=tue.default.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2gb
#SBATCH --time=03:00:00
#SBATCH --output=hpc/logs/data_gen_%j.out
#SBATCH --error=hpc/logs/data_gen_%j.err

# Run from the project root:
#   sbatch hpc/run_data_generation.sh

set -euo pipefail

module purge
module load Python/3.11.3-GCCcore-12.3.0

source ~/ctc_env/bin/activate

# Numba uses all available CPUs by default; pin it to the Slurm allocation so
# it doesn't steal cores from other jobs on the same node.
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Using $NUMBA_NUM_THREADS threads (SLURM_CPUS_PER_TASK)"
echo "Working directory: $(pwd)"

python ctc_adjusted/ctc_h2_multiple_collisions_numba.py
