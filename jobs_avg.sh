#!/bin/bash
#SBATCH --job-name=saraB
#SBATCH --nodes=1
#SBATCH --time=5-00:00:00
#SBATCH --partition=mech-cst-mov.cpu.q
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=128G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=s.betancur.giraldo@student.tue.nl

module load matlab/2023b
module load Gurobi/11.0.0-GCCcore-12.3.0
matlab -nodesktop -r "run_results_avgAcc;quit"

