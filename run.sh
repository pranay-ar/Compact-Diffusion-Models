#!/bin/bash
#SBATCH -c 12  # Number of Cores per Task
#SBATCH --mem=200000  # Requested Memory
#SBATCH -p gypsum-m40  # Partition
#SBATCH --gres=gpu:4  # Number of GPUs
#SBATCH -t 72:00:00  # Job time limit
#SBATCH -o slurm_jobs/slurm-%j.out  # %j = job ID
#SBATCH --mail-type=END

source /home/pranayr_umass_edu/miniconda3/etc/profile.d/conda.sh
conda activate imu2clip

python ddpm_conditional.py --config ./configs/compressed.yaml
