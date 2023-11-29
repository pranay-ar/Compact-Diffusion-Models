import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
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

#to-do
def compute_fid(original_images, generated_images, device):
    pass
