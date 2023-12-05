import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from ddpm_conditional import Diffusion
from pytorch_fid import fid_score
import time


def plot_images(images):
    plt.figure(figsize=(64, 64))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    for i, image in enumerate(images):
        ndarr = image.permute(1, 2, 0).to('cpu').numpy()
        im = Image.fromarray(ndarr)
        im.save(os.path.join(path, f"{i}.jpg"))
    
    print("Images have been saved in the folder:", path)

def make_grid(images,path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(configs):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # configs.image_size + 1/4 *configs.image_size
        torchvision.transforms.RandomResizedCrop(configs.get("image_size"), scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(configs.get("dataset_path"), transform=transforms)
    dataloader = DataLoader(dataset, batch_size=configs.get("batch_size"), shuffle=True)
    return dataloader


def setup_logging(run_name):
    model_dir = os.path.join("models", run_name)
    results_dir = os.path.join("results", run_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    return model_dir, results_dir

def calculate_fid(model, results_dir, dataset_path, device):
    # time it
    start = time.time()
    fid = fid_score.calculate_fid_given_paths([dataset_path, results_dir],
                                              batch_size=64,  # Adjust batch size to your hardware
                                              device=device,
                                              dims=2048)  # Inception features dimension
    end = time.time()
    print(f'Time taken: {end - start} seconds')
    print(f'FID score: {fid}')
    return fid

def fix_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # remove module. prefix if present
        new_state_dict[name] = v
    return new_state_dict
