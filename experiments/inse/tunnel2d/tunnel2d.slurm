#!/bin/bash
#SBATCH --job-name=circ_evol
#SBATCH --account=bsc21
#SBATCH --chdir=.
#SBATCH --output=experiments/inse/tunnel2d/circle_data/output.out
#SBATCH --error=experiments/inse/tunnel2d/circle_data/error.err
#SBATCH --ntasks=1                    
#SBATCH --cpus-per-task=8           
#SBATCH --time=06:00:00               
#SBATCH --qos=gp_bsccase       

# Load necessary modules (Adjust version as needed)
module load julia/1.12.0

# Optional: Print configuration to output for verification
echo "------------------------------------------------"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $HOSTNAME"
echo "------------------------------------------------"

julia --project=. -t 1 experiments/inse/tunnel2d/tunnel2d.jl
