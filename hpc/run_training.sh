#!/bin/bash
#SBATCH --job-name=mdn_train
#SBATCH --partition=tue.default.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4gb
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=hpc/logs/train_%j.out
#SBATCH --error=hpc/logs/train_%j.err

# Run from the project root:
#   sbatch hpc/run_training.sh
#
# To request a specific GPU (e.g. A30), add to the directives above:
#   #SBATCH --constraint="a30"
#
# To run without a GPU (CPU-only fallback), remove the --gpus line.
# The training code detects CUDA automatically.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — edit these before submitting
# ---------------------------------------------------------------------------
DATASET="data/H2H2_collisions_numba_b1_0_Etr20k_Erot15k_1000000_seed42.npy"
KIND="mdn"                          # mdn | beta_mdn
EPOCHS=100
BATCH_SIZE=512                      # larger batches exploit GPU parallelism
LR=2e-4
PATIENCE=30
WF="None"                           # polynomial weight exponent, or None for unweighted
OUTPUT="results/models/mdn/mdn_H2_Etr20k_Erot15k.pth"
# ---------------------------------------------------------------------------

module purge
module load Python/3.11.3-GCCcore-12.3.0

source ~/ctc_env/bin/activate

echo "Working directory: $(pwd)"
echo "Dataset:   $DATASET"
echo "Output:    $OUTPUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU detected — running on CPU"

python -c "
from training.core import train_collision_model
train_collision_model(
    kind='$KIND',
    datapath='$DATASET',
    outputpath='$OUTPUT',
    epochs=$EPOCHS,
    batch_size=$BATCH_SIZE,
    lr=$LR,
    wf=$WF,
    patience=$PATIENCE,
)
"
