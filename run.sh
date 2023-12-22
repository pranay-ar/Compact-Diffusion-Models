#!/bin/bash
#SBATCH -c 12  # Number of Cores per Task
#SBATCH --mem=200000  # Requested Memory
#SBATCH -p gypsum-m40  # Partition
#SBATCH --gres=gpu:4  # Number of GPUs
#SBATCH -t 72:00:00  # Job time limit
#SBATCH -o slurm_jobs/slurm-%j.out  # %j = job ID
#SBATCH --mail-type=END

source /home/pranayr_umass_edu/miniconda3/etc/profile.d/conda.sh
conda activate cdm

python generate.py
# python ddpm_conditional.py --config "./configs/fine_tune.yaml"

# python ddpm_prune.py \
# --dataset ./data/cifar10-64/train/ \
# --model_path ./models/DDPM_conditional/ema_ckpt.pt \
# --save_path ./models/pruned/ddpm_conditional_pruned \
# --pruning_ratio 0.16 \
# --batch_size 32 \
# --pruner diff-pruning \
# --thr 0.05 \
# --device cuda \