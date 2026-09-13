#!/bin/bash
#SBATCH -A hpc1906185151
#SBATCH --partition=C064M1024G
#SBATCH --qos=low
#SBATCH -J wm2-job-20260329-zephyr
#SBATCH --nodes=6
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=7
#SBATCH --time=7200
#SBATCH --chdir=/lustre/home/2501110202/work/ScarToGriffith
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err

set -euo pipefail

module load julia

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
# 不要再加 srun：SlurmClusterManager 会自行启动 worker。
julia --startup-file=no --threads=2 random_ising/run_slurm.jl
