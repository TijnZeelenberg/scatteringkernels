#!/bin/bash
#SBATCH --job-name=lammps_h2relax
#SBATCH --partition=tue.default.q
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2gb
#SBATCH --time=04:00:00
#SBATCH --output=hpc/logs/lammps_h2relax_%j.out
#SBATCH --error=hpc/logs/lammps_h2relax_%j.err

# Run from the project root:
#   sbatch hpc/run_lammps_h2relaxation.sh
#
# Notes:
#   - rigid/small does not support the OMP suffix, so pure MPI is used.
#   - 8 MPI ranks is a reasonable balance for 20 000 molecules in a 100 nm box;
#     beyond ~16 ranks, inter-rank communication dominates for this system size.
#   - Estimated walltime: ~2 h at 8 ranks; 4 h gives comfortable headroom.

set -euo pipefail

module purge
module load LAMMPS/29Aug2024_update2-foss-2023a-kokkos

echo "Working directory: $(pwd)"
echo "LAMMPS binary:     $(which lmp)"
echo "MPI tasks:         $SLURM_NTASKS"
echo "Node:              $SLURMD_NODENAME"

# The input file references h2_init.data and writes to output/ using relative
# paths, so we run from the lammps/ sub-directory.
cd lammps
mkdir -p output

mpirun -np "$SLURM_NTASKS" lmp -in in.h2relaxation
