#!/bin/bash
#SBATCH --job-name=h2_lammps_relax
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

# Set environment
module purge
module load LAMMPS/29Aug2024_update2-foss-2024a-kokkos

echo "Working directory: $(pwd)"

# Run from lammps/ so relative paths in the input file (h2_init.data, output/) resolve correctly
cd lammps
lmp < in.h2relaxation
