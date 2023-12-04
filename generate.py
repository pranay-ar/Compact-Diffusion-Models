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
from ddpm_conditional import Diffusion

device = "cuda"
model = UNet_conditional(num_classes=10).to(device)
# model = nn.DataParallel(model)
ckpt = torch.load("/work/pi_adrozdov_umass_edu/pranayr_umass_edu/cs682/Diffusion-Models-pytorch/models/DDPM_conditional/ema_ckpt.pt")
ckpt = fix_state_dict(ckpt)
model.load_state_dict(ckpt)
diffusion = Diffusion(img_size=64, device=device)
total_images_per_class = 1024
batch_size = 256
cfg_scale = 0

for class_index in range(10):  # Assuming 10 classes
    class_folder = f"./generated_images/class_{class_index}"
    os.makedirs(class_folder, exist_ok=True)
    
    y = torch.full((total_images_per_class,), class_index, dtype=torch.long).to(device)
    x = diffusion.sample(model, total_images_per_class, batch_size, y, cfg_scale=cfg_scale)
    
    # Assuming save_images function handles saving all generated images for a class
    save_images(x, class_folder) 

    print(f"Images for class {class_index} have been saved in {class_folder}.")

# results_dir = "./generated_images"  # Directory where generated images will be saved
# dataset_path = './data/cifar10-64'  # Path to your CIFAR10 dataset
# device = "cuda"

# # Call the function to generate images and calculate FID
# fid_score = calculate_fid(model, results_dir, dataset_path, device)