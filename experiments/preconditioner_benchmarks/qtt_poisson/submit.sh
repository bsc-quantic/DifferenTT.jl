#!/bin/bash
#SBATCH --job-name=qtt_poisson
#SBATCH --output=logs/qtt_poisson_%A_%a.out
#SBATCH --error=logs/qtt_poisson_%A_%a.err
#SBATCH --array=1-105
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

mkdir -p logs
julia --project=. experiments/preconditioner_benchmarks/qtt_poisson/worker.jl $SLURM_ARRAY_TASK_ID
