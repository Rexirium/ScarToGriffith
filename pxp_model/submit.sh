#!/bin/bash
#SBATCH -A hpc1906185151
#SBATCH --partition=C064M1024G
#SBATCH --qos=low
#SBATCH -J pxp-scan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --time=7200
#SBATCH --chdir=/lustre/home/2501110202/work/ScarToGriffith
#SBATCH --output=pxp_scan.%j.out
#SBATCH --error=pxp_scan.%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Quantum

workers=7
threads=8
export OMP_NUM_THREADS="$threads"
export MKL_NUM_THREADS="$threads"
export OPENBLAS_NUM_THREADS="$threads"
export BLIS_NUM_THREADS="$threads"
export OMP_MAX_ACTIVE_LEVELS=1

# One Slurm task owns all CPUs; Python spawns the seven L workers locally.
# Submit: sbatch pxp_model/submit.sh [--output-dir path]
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=none \
    python -u pxp_model/pxp_scan.py --workers "$workers" --threads "$threads" "$@"
