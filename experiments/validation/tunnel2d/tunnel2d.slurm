#!/bin/bash
#SBATCH --job-name=circ_evol
# Static #SBATCH lines do not expand variables: override the account and QoS with
#   sbatch --account=<project> --qos=<qos> validation/tunnel2d/tunnel2d.slurm
#SBATCH --account=bsc21
#SBATCH --chdir=.
#SBATCH --output=experiments/validation/tunnel2d/circle_data/output.out
#SBATCH --error=experiments/validation/tunnel2d/circle_data/error.err
#SBATCH --ntasks=1                    
#SBATCH --cpus-per-task=8           
#SBATCH --time=06:00:00               
#SBATCH --qos=gp_bsccase       

# Load necessary modules (Adjust version as needed)
module load ${JULIA_MODULE:-julia/1.12.0}

# Optional: Print configuration to output for verification
echo "------------------------------------------------"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $HOSTNAME"
echo "------------------------------------------------"

julia --project=experiments -t 1 --heap-size-hint=14G experiments/validation/tunnel2d/tunnel2d.jl
