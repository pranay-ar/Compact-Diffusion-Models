import torch
from torchvision.utils import save_image
from ddpm import Diffusion
from utils import get_data
import argparse

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.batch_size = 1  # 5
args.image_size = 64
args.dataset_path = r"/work/pi_adrozdov_umass_edu/pranayr_umass_edu/cs682/Diffusion-Models-pytorch/cifar10-64/test/"

dataloader = get_data(args)

diff = Diffusion(device="cpu")

image = next(iter(dataloader))[0]
t = torch.Tensor([50, 100, 150, 200, 300, 600, 700, 999]).long()

noised_image, _ = diff.noise_images(image, t)
save_image(noised_image.add(1).mul(0.5), "noise.jpg")
