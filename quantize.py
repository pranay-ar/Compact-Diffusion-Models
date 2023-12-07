import torch
import torch.nn as nn
from tqdm import tqdm
from utils import *
from modules import UNet_conditional, EMA
from ddpm_conditional import Diffusion
import torch.quantization

model = UNet_conditional(compress=1,num_classes=10)
ckpt = torch.load("./models/DDPM_conditional/ema_ckpt.pt")
ckpt = fix_state_dict(ckpt)
model.load_state_dict(ckpt)
model.eval()

quantize_layers = {nn.Conv2d, nn.Linear}

print("Before quantization:")
print_size_of_model(model)

model_quantized = torch.quantization.quantize_dynamic(
    model,
    quantize_layers,
    dtype=torch.qint8
)

print("After quantization:")
print_size_of_model(model_quantized)

torch.save(model_quantized.state_dict(), "./models/DDPM_conditional/ema_ckpt_quantized.pt")
