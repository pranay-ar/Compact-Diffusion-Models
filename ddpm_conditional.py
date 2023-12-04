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
from torch.cuda.amp import GradScaler, autocast

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train(configs):

    wandb.init(project="CDM", config=configs, name="DDPM_conditional") if configs.get("wandb", True) else wandb.init(mode="disabled")

    logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
    
    #load model parameters

    device = configs.get("device", "cuda")
    batch_size = configs.get("batch_size", 64)
    epochs = configs.get("epochs", 300)
    lr = configs.get("lr", 3e-4)
    num_classes = configs.get("num_classes", 10)
    image_size = configs.get("image_size", 64)
    run_name = configs.get("run_name")
    model_dir, results_dir = setup_logging(run_name) # will distillation reflect in path
    
    use_distillation = configs.get("distillation", False)
    compressed_model = configs.get("compress", False)

    setup_logging(run_name)
    dataloader = get_data(configs)
    
    model = UNet_conditional(compress=2 if compressed_model else 1,
                             num_classes=num_classes).to(device)
    model = torch.nn.DataParallel(model)
    if use_distillation:
        teacher = UNet_conditional(num_classes=num_classes).to(device)
        teacher = torch.nn.DataParallel(teacher)
        teacher.load_state_dict(args.teacher_path)
        teacher.eval()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=image_size, device=device)
    l = len(dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    use_mixed_precision = configs.get("mixed_precision", False)
    scaler = GradScaler() if use_mixed_precision else None
    print(f"Using mixed precision: {use_mixed_precision}")

    for epoch in range(epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)

            optimizer.zero_grad()

            if use_mixed_precision:
                with autocast():
                    x_t, noise = diffusion.noise_images(images, t)
                    if np.random.random() < 0.1:
                        labels = None
                    predicted_noise = model(x_t, t, labels)
                    if use_distillation:
                        loss = mse(teacher(x_t, t), predicted_noise) # can try weighted sum
                    else:
                        loss = mse(noise, predicted_noise)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                x_t, noise = diffusion.noise_images(images, t)
                if np.random.random() < 0.1:
                    labels = None
                predicted_noise = model(x_t, t, labels)
                loss = mse(noise, predicted_noise)
                loss.backward()
                optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            wandb.log({"MSE": loss.item(), "Epoch": epoch, "Batch": i})

        if epoch % 50 == 0:
            labels = torch.arange(10).long().to(device)
            sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
            ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
            plot_images(sampled_images)
            save_images(sampled_images, os.path.join(results_dir, f"{epoch}.jpg"))
            save_images(ema_sampled_images, os.path.join(results_dir, f"{epoch}_ema.jpg"))
            torch.save(model.module.state_dict(), os.path.join(model_dir, f"ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join(model_dir, f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join(model_dir, f"optim.pt"))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
            "--config", default="./configs/default.yaml"
        )
    parser.add_argument(
            "--teacher_path", default=None # make it part of config/run_name?
        )
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    
    print(configs)
    train(configs)