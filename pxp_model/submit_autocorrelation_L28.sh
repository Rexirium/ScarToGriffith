#!/bin/bash
#SBATCH -A hpc1906185151
#SBATCH --partition=C064M1024G
#SBATCH --qos=low
#SBATCH --job-name=pxp-autocorrelation-L28
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=7200
#SBATCH --chdir=/lustre/home/2501110202/work/ScarToGriffith
#SBATCH --output=pxp_autocorrelation.L28.%j.out
#SBATCH --error=pxp_autocorrelation.L28.%j.err

set -euo pipefail

# Submit only L=28: sbatch pxp_model/submit_autocorrelation_L28.sh
# The TOML sets 2 workers x 32 threads and a 960 GiB/node memory budget.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Quantum

# All ranks share this ID; the job ID separates simultaneous submissions.
run_id="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}_L28"
srun --mpi=none --ntasks="$SLURM_NTASKS" --ntasks-per-node=1 \
    --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=cores --kill-on-bad-exit=1 \
    --output='pxp_autocorrelation.L28.%j.rank%t.out' \
    --error='pxp_autocorrelation.L28.%j.rank%t.err' \
    python -u pxp_model/pxp_autocorrelation_scan.py \
    --config pxp_model/pxp_autocorrelation_scan_L28.toml --run-id "$run_id"
