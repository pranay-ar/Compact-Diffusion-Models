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
# model = UNet_conditional(compress=2, num_classes=10).to(device)
model = torch.load("./models/pruned/ddpm_conditional_pruned/pruned/unet_pruned_0.16_0.01.pth")
ckpt = torch.load("./models/Pruned_0.16_0.01_FT/ema_ckpt.pt")
ckpt = fix_state_dict(ckpt)
model.load_state_dict(ckpt, strict=False)
diffusion = Diffusion(img_size=64, device=device)
total_images_per_class = 1024
batch_size = 256
cfg_scale = 0

for class_index in range(10):  # Assuming 10 classes
    class_folder = f"./fid_data/generated_images_pruned_0.16_0.01_FT/class_{class_index}"
    os.makedirs(class_folder, exist_ok=True)
    
    y = torch.full((total_images_per_class,), class_index, dtype=torch.long).to(device)
    x = diffusion.sample(model, total_images_per_class, batch_size, y, cfg_scale=cfg_scale)
    
    # Assuming save_images function handles saving all generated images for a class
    save_images(x, class_folder) 

    print(f"Images for class {class_index} have been saved in {class_folder}.")

# results_dir = "./fid_data/combined_generated_images_KD"  # Directory where generated images will be saved
# dataset_path = './fid_data/combined_real_train'  # Path to your CIFAR10 dataset
# device = "cuda"

# # Call the function to generate images and calculate FID
# fid_score = calculate_fid(results_dir, dataset_path, device)