#!/bin/sh

#SBATCH --account=theory
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16gb
#SBATCH --time=5-00:00:00
#SBATCH --array=0-6%12
#SBATCH --job-name=Movie_experiments_name
#SBATCH --output=slurm/slurm_%x_%a_%A.out
#SBATCH --error=slurm/slurm_%x_%a_%A.err 

[[ ! -d slurm ]] && mkdir slurm
experiment_name=$(sed -n "${SLURM_ARRAY_TASK_ID}p" experiments.txt)
echo Queueing experiment $experiment_name
source ~/.bashrc
conda activate pytorch_env
srun python Main.py $experiment_name
conda deactivate
