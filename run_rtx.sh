#!/bin/bash
#SBATCH -c 12  # Number of Cores per Task
#SBATCH --mem=200000  # Requested Memory
#SBATCH -p gypsum-2080ti  # Partition
#SBATCH --gres=gpu:8  # Number of GPUs
#SBATCH -t 72:00:00  # Job time limit
#SBATCH -o slurm_jobs/slurm-%j.out  # %j = job ID
#SBATCH --mail-type=END

# source /home/pranayr_umass_edu/miniconda3/etc/profile.d/conda.sh
# conda activate imu2clip

python ddpm_conditional.py --config "/work/pi_adrozdov_umass_edu/pranayr_umass_edu/cs682/Diffusion-Models-pytorch/configs/rtx.yaml"