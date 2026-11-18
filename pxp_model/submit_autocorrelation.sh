#!/bin/bash
#SBATCH -A hpc1906185151
#SBATCH --partition=C064M1024G
#SBATCH --qos=low
#SBATCH --job-name=pxp-autocorrelation
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=7200
#SBATCH --chdir=/lustre/home/2501110202/work/ScarToGriffith
#SBATCH --output=pxp_autocorrelation.%j.out
#SBATCH --error=pxp_autocorrelation.%j.err

set -euo pipefail

# Account, partition, environment and work directory follow submit.sh.
# Adjust them and the memory request for your cluster/allocation as needed.
# Submit both scans in order: sbatch pxp_model/submit_autocorrelation.sh
# Both scans share the same 8-node allocation and total walltime above.
# Optional single config: sbatch pxp_model/submit_autocorrelation.sh --config /shared/scan.toml
# The scanner checks individual points and limits concurrent estimated memory.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Quantum

# Generate once in the batch shell so every rank uses the same directory name.
# The job ID prevents collisions when separate scans start in the same second.
run_id="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"

run_scan() {
    local stage="$1"
    shift
    srun --mpi=none --ntasks="$SLURM_NTASKS" --ntasks-per-node=1 \
        --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=cores --kill-on-bad-exit=1 \
        --output="pxp_autocorrelation.${stage}.%j.rank%t.out" \
        --error="pxp_autocorrelation.${stage}.%j.rank%t.err" \
        python -u pxp_model/pxp_autocorrelation_scan.py "$@" --run-id "${run_id}_${stage}"
}

if (( $# )); then
    run_scan custom "$@"
else
    run_scan Lle26 --config pxp_model/pxp_autocorrelation_scan.toml
    run_scan L28 --config pxp_model/pxp_autocorrelation_scan_L28.toml
fi
