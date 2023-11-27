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

def fix_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # remove module. prefix if present
        new_state_dict[name] = v
    return new_state_dict

device = "cuda"
model = UNet_conditional(num_classes=10).to(device)
ckpt = torch.load("./models/DDPM_conditional/ckpt.pt")
ckpt = fix_state_dict(ckpt)
model.load_state_dict(ckpt)
diffusion = Diffusion(img_size=64, device=device)
n = 8
y = torch.Tensor([6] * n).long().to(device)
x = diffusion.sample(model, n, y, cfg_scale=0)
plot_images(x)