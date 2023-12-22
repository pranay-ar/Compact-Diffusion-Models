import torch
from torchvision.utils import save_image
from ddpm import Diffusion
from utils import get_data


batch_size = 1
image_size = 64
dataset_path = r"./data/cifar10/test/"
configs = {
    "batch_size": batch_size,
    "image_size": image_size,
    "dataset_path": dataset_path
}


dataloader = get_data(configs)

diff = Diffusion(device="cpu")

image = next(iter(dataloader))[0]
t = torch.Tensor([50, 100, 150, 200, 300, 600, 700, 999]).long()

noised_image, _ = diff.noise_images(image, t)
save_image(noised_image.add(1).mul(0.5), "noise.jpg")
