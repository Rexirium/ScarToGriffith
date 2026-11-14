#!/bin/bash
#SBATCH -A hpc1906185151
#SBATCH --partition=C064M1024G
#SBATCH --qos=low
#SBATCH --job-name=pxp-autocorrelation
#SBATCH --nodes=6
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=7200
#SBATCH --chdir=/lustre/home/2501110202/work/ScarToGriffith
#SBATCH --output=pxp_autocorrelation.%j.out
#SBATCH --error=pxp_autocorrelation.%j.err

set -euo pipefail

# Account, partition, environment and work directory follow submit.sh.
# Adjust them and the memory request for your cluster/allocation as needed.
# Submit from the repository: sbatch pxp_model/submit_autocorrelation.sh
# Optional config: sbatch pxp_model/submit_autocorrelation.sh --config /shared/scan.toml
# Request the measured per-node budget with sbatch --mem=... when needed;
# Python checks that allocation before ED. --mem=0 needs TOML memory_gib.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Quantum

# Generate once in the batch shell so every rank uses the same directory name.
run_id="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S_%N)_job${SLURM_JOB_ID}"
srun --mpi=none --ntasks="$SLURM_NTASKS" --ntasks-per-node=1 \
    --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=cores --kill-on-bad-exit=1 \
    --output='pxp_autocorrelation.%j.rank%t.out' \
    --error='pxp_autocorrelation.%j.rank%t.err' \
    python -u pxp_model/pxp_autocorrelation_scan.py "$@" --run-id "$run_id"
