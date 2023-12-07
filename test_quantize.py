import torch
import torch.nn as nn
from utils import *
from modules import UNet_conditional

# # Initialize the model
# model = UNet_conditional(compress=1, num_classes=10)
# model.eval()

# # Load the pre-trained model weights
# ckpt = torch.load("./models/DDPM_conditional/ema_ckpt.pt")
# ckpt = fix_state_dict(ckpt)
# model.load_state_dict(ckpt)

# # Specify the layers to be quantized
# quantize_layers = {nn.Conv2d, nn.Linear}

# # Apply dynamic quantization
# model_quantized = torch.quantization.quantize_dynamic(
#     model, quantize_layers, dtype=torch.qint8
# )

# # Save the entire quantized model
# torch.save(model_quantized, "./models/DDPM_conditional/ema_ckpt_quantized.pth")

import torch
from ddpm_conditional import Diffusion
from torchvision.utils import save_image

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