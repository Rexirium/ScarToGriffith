#!/bin/bash
#SBATCH -A hpc1906185151
#SBATCH --partition=C064M1024G
#SBATCH --qos=low
#SBATCH --job-name=pxp-autocorrelation-Lle26
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=7200
#SBATCH --chdir=/lustre/home/2501110202/work/ScarToGriffith
#SBATCH --output=pxp_autocorrelation.Lle26.%j.out
#SBATCH --error=pxp_autocorrelation.Lle26.%j.err

set -euo pipefail

# Submit only L<=26: sbatch pxp_model/submit_autocorrelation_Lle26.sh
# Memory scheduling uses Slurm's numeric allocation or memory_gib in the TOML.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Quantum

# All ranks share this ID; the job ID separates simultaneous submissions.
run_id="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}_Lle26"
srun --mpi=none --ntasks="$SLURM_NTASKS" --ntasks-per-node=1 \
    --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=cores --kill-on-bad-exit=1 \
    --output='pxp_autocorrelation.Lle26.%j.rank%t.out' \
    --error='pxp_autocorrelation.Lle26.%j.rank%t.err' \
    python -u pxp_model/pxp_autocorrelation_scan.py \
    --config pxp_model/pxp_autocorrelation_scan.toml --run-id "$run_id"
