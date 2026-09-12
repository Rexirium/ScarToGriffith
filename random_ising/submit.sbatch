#!/bin/bash
# 从仓库根目录提交：sbatch random_ising/submit.sbatch
#SBATCH --job-name=rbim-scan
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --output=rbim-%j.out
#SBATCH --error=rbim-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
# 按服务器环境加载 Julia，例如 module load julia/1.12。
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
# 不要再加 srun：SlurmClusterManager 会自行启动 worker。
julia --startup-file=no --threads=1 random_ising/run_slurm.jl
