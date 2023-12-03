import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from argparse import ArgumentParser
from modules import UNet_conditional, EMA
import logging
import wandb
import yaml
from ddpm_conditional import Diffusion

device = "cuda"
model = UNet_conditional(num_classes=10).to(device)
# model = nn.DataParallel(model)
ckpt = torch.load("/work/pi_adrozdov_umass_edu/pranayr_umass_edu/cs682/Diffusion-Models-pytorch/models/DDPM_conditional/ema_ckpt.pt")
ckpt = fix_state_dict(ckpt)
model.load_state_dict(ckpt)
# diffusion = Diffusion(img_size=64, device=device)
# n = 8
# y = torch.Tensor([5] * n).long().to(device)
# x = diffusion.sample(model, n, y, cfg_scale=0)
# plot_images(x)
# save_images(x,"./images.png")
# print("Images have been saved.")

epoch = 0  # Since it's inference, epoch is not relevant, but needed for the function
results_dir = "./generated_images"  # Directory where generated images will be saved
dataset_path = './data/cifar10-64'  # Path to your CIFAR10 dataset
device = "cuda"

# Call the function to generate images and calculate FID
fid_score = calculate_fid_for_epoch(model, epoch, results_dir, dataset_path, device)
print(f"FID score: {fid_score}")