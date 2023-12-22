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

device = "cpu"
# model = torch.load("./models/pruned/ddpm_conditional_pruned/pruned/unet_pruned_0.16_0.01.pth")
model = UNet_conditional(compress=1,num_classes=10).to(device)
ckpt = torch.load("./models/DDPM_conditional/ema_ckpt.pt", map_location=device)
ckpt = fix_state_dict(ckpt)
model.load_state_dict(ckpt)
diffusion = Diffusion(img_size=64, device=device)
n = 1
y = torch.Tensor([6] * n).long().to(device)
x = diffusion.sample(model, 1, 1, labels=y, cfg_scale=0)
# make_grid(x, "./test.jpg")