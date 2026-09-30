#!/bin/bash
#SBATCH -A hpc1906185151
#SBATCH --partition=C064M0256G
#SBATCH --qos=low
#SBATCH -J random-tfim
#SBATCH --nodes=4
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=7
#SBATCH --time=7200
#SBATCH --chdir=/lustre/home/2501110202/work/ScarToGriffith
#SBATCH --output=random_tfim.%j.out
#SBATCH --error=random_tfim.%j.err

set -euo pipefail

module load julia

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
# Args: [demo|full] [samples] [output.h5] [open|periodic] [uniform|fixed]
# Example: sbatch random_tfim/submit.sh full 50_000 random_tfim/results/full_fixed.h5 periodic fixed
# SlurmClusterManager starts the worker processes; do not add srun here.
if (($# == 0)); then
	set -- full 50_000
fi
julia --startup-file=no --threads=2 random_tfim/run_slurm.jl "$@"
