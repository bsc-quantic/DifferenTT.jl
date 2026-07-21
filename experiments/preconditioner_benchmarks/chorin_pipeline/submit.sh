#!/bin/bash
#SBATCH --job-name=chorin_pipeline
#SBATCH --output=logs/chorin_pipeline_%A_%a.out
#SBATCH --error=logs/chorin_pipeline_%A_%a.err
#SBATCH --array=1-135
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

mkdir -p logs
julia --project=. experiments/preconditioner_benchmarks/chorin_pipeline/worker.jl $SLURM_ARRAY_TASK_ID
