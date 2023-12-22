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

# Load the entire quantized model
device = "cpu"
quantized_model = torch.load("./models/DDPM_conditional/ema_ckpt_quantized.pth", map_location=device)

# Prepare the model for inference
quantized_model.eval()

# Initialize the diffusion process
diffusion = Diffusion(img_size=64, device=device)

# Sample data for inference
n = 1
y = torch.Tensor([6] * n).long().to(device)

# Generate sample using the quantized model
with torch.no_grad():
    outputs = diffusion.sample(quantized_model, n, 1,labels=y, cfg_scale=0)
    # Assuming outputs is a list, select the first tensor
    x = outputs[0] if isinstance(outputs, list) else outputs

    # Normalize the output to [0, 1] range
    x = (x - x.min()) / (x.max() - x.min())

# Save the output image
save_image(x, "./test_quantized.jpg")
