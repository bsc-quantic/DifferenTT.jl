#!/bin/bash
#SBATCH --job-name=tt_poisson
#SBATCH --output=logs/tt_poisson_%A_%a.out
#SBATCH --error=logs/tt_poisson_%A_%a.err
#SBATCH --array=1-64
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

mkdir -p logs
julia --project=. experiments/preconditioner_benchmarks/tt_poisson/worker.jl $SLURM_ARRAY_TASK_ID
