#!/bin/bash
#SBATCH --job-name=o2_lammps_relax
#SBATCH --partition=mech.pf-student.q        # verify with: sinfo
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2gb
#SBATCH --time=04:00:00
#SBATCH --output=hpc/logs/%x%j.out
#SBATCH --error=hpc/logs/%x%j.err
#SBATCH --chdir=/home/20193567/scatteringkernels

set -euo pipefail

module purge
module load LAMMPS/29Aug2024_update2-foss-2024a-kokkos

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

echo "Working directory: $(pwd)"
echo "MPI tasks: $SLURM_NTASKS, threads/task: $SLURM_CPUS_PER_TASK"

cd lammps
mpirun -np "$SLURM_NTASKS" lmp \
    -k on t "$SLURM_CPUS_PER_TASK" \
    -sf kk \
    -pk kokkos newton on neigh half \
    < in.o2relaxation
