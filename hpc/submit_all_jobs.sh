#!/bin/bash
# Master job submission script that handles dependencies between H2 CTC and model training jobs

set -euo pipefail

echo "Submitting H2 CTC batch size job..."
JOB_H2_BATCHSIZE=$(sbatch hpc/run_h2_ctc_batchsize.sh | awk '{print $NF}')
echo "Submitted H2 CTC batch size job with ID: $JOB_H2_BATCHSIZE"

echo "Submitting H2 CTC impact parameter job..."
JOB_H2_IMPACTPARAM=$(sbatch hpc/run_h2_ctc_impactparam.sh | awk '{print $NF}')
echo "Submitted H2 CTC impact parameter job with ID: $JOB_H2_IMPACTPARAM"

echo ""
echo "Submitting model training batch size job (depends on H2 CTC batch size)..."
JOB_MODEL_BATCHSIZE=$(sbatch --dependency=afterok:"$JOB_H2_BATCHSIZE" hpc/run_modeltraining_batchsize.sh | awk '{print $NF}')
echo "Submitted model training batch size job with ID: $JOB_MODEL_BATCHSIZE"

echo "Submitting model training impact parameter job (depends on H2 CTC impact parameter)..."
JOB_MODEL_IMPACTPARAM=$(sbatch --dependency=afterok:"$JOB_H2_IMPACTPARAM" hpc/run_modeltraining_impactparam.sh | awk '{print $NF}')
echo "Submitted model training impact parameter job with ID: $JOB_MODEL_IMPACTPARAM"

echo ""
echo "Job submission complete:"
echo "  H2 CTC batch size: $JOB_H2_BATCHSIZE"
echo "  H2 CTC impact param: $JOB_H2_IMPACTPARAM"
echo "  Model training batch size (depends on $JOB_H2_BATCHSIZE): $JOB_MODEL_BATCHSIZE"
echo "  Model training impact param (depends on $JOB_H2_IMPACTPARAM): $JOB_MODEL_IMPACTPARAM"
