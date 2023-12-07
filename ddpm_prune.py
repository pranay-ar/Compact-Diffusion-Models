
import torch_pruning as tp
import torch
import torchvision
from torchvision import transforms
import torchvision
from tqdm import tqdm
import os
from glob import glob
from PIL import Image
from utils import fix_state_dict, get_dataset

import numpy as np
import torch.nn as nn
from modules import UNet_conditional, EMA
from ddpm_conditional import Diffusion

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str,  default=None, help="path to an image folder")
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--pruning_ratio", type=float, default=0.3)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--pruner", type=str, default='taylor', choices=['taylor', 'random', 'magnitude', 'reinit', 'diff-pruning'])

parser.add_argument("--thr", type=float, default=0.05, help="threshold for diff-pruning")

args = parser.parse_args()

batch_size = args.batch_size
dataset = args.dataset

if __name__=='__main__':
    
    # loading images for gradient-based pruning
    if args.pruner in ['taylor', 'diff-pruning']:
        dataset = get_dataset(args.dataset)
        print(f"Dataset size: {len(dataset)}")
        train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, drop_last=True
        )
        import torch_pruning as tp
        clean_images = next(iter(train_dataloader))
        if isinstance(clean_images, (list, tuple)):
            clean_images, labels = clean_images
            labels = labels.to(args.device)
        clean_images = clean_images.to(args.device)
        noise = torch.randn(clean_images.shape).to(clean_images.device)

    # Loading pretrained model
    print("Loading pretrained model from {}".format(args.model_path))
    
    model = UNet_conditional(num_classes=10).to(args.device)
    ckpt = torch.load(args.model_path)
    ckpt = fix_state_dict(ckpt)
    model.load_state_dict(ckpt)
    model = torch.nn.DataParallel(model)
    diffusion = Diffusion(img_size=64, device=args.device)

    model = model.eval()
    if 'cifar' in args.dataset:
        print("############")
        example_inputs = {'x': torch.randn(1, 3, 64, 64).to(args.device), 't': torch.ones((1,)).long().to(args.device), 'y': None}
    else:
        example_inputs = {'x': torch.randn(1, 3, 256, 256).to(args.device), 't': torch.ones((1,)).long().to(args.device), 'y': None}

    if args.pruning_ratio>0:
        if args.pruner == 'taylor':
            imp = tp.importance.TaylorImportance(multivariable=True) # standard first-order taylor expansion
        elif args.pruner == 'random' or args.pruner=='reinit':
            imp = tp.importance.RandomImportance()
        elif args.pruner == 'magnitude':
            imp = tp.importance.MagnitudeImportance()
        elif args.pruner == 'diff-pruning':
            imp = tp.importance.TaylorImportance(multivariable=False) # a modified version, estimating the accumulated error of weight removal
        else:
            raise NotImplementedError

        ignored_layers = [model.module.outc]
        channel_groups = {}
        #from diffusers.models.attention import 
        #for m in model.modules():
        #    if isinstance(m, AttentionBlock):
        #        channel_groups[m.query] = m.num_heads
        #        channel_groups[m.key] = m.num_heads
        #        channel_groups[m.value] = m.num_heads
        
        pruner = tp.pruner.MagnitudePruner(
            model.module,
            example_inputs,
            importance=imp,
            iterative_steps=1,
            channel_groups=channel_groups,
            ch_sparsity=args.pruning_ratio,
            ignored_layers=ignored_layers,
            round_to = 16
        )

        base_macs, base_params = tp.utils.count_ops_and_params(model.module, example_inputs)
        model.zero_grad()
        model.eval()
        import random
        mse = nn.MSELoss()
        if args.pruner in ['taylor', 'diff-pruning']:
            loss_max = 0

            print("Accumulating gradients for pruning...")
            for step_k in tqdm(range(1000)):
                timesteps = (step_k*torch.ones((args.batch_size,), device=clean_images.device)).long()
                x_t, noise = diffusion.noise_images(clean_images, timesteps)
                if np.random.random() < 0.1:
                    labels = None
                predicted_noise = model(x_t, timesteps, labels)
                loss = mse(noise, predicted_noise) / 8
                loss.backward()
                
                if args.pruner=='diff-pruning':
                    if loss>loss_max: loss_max = loss
                    if loss<loss_max * args.thr: break # taylor expansion over pruned timesteps ( L_t / L_max > thr )
        
        for g in pruner.step(interactive=True):
            g.prune()
        
        # DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)
        # for (target_module, pruning_fn, idxs) in pruning_record:
        #     group = DG.get_pruning_group(target_module, pruning_fn, idxs)
        # Update static attributes
        # from diffusers.models.resnet import Upsample2D, Downsample2D
        # for m in model.modules():
        #     if isinstance(m, (Upsample2D, Downsample2D)):
        #         m.channels = m.conv.in_channels
        #         m.out_channels == m.conv.out_channels
        print("####printing model #########")
        print(model)
        macs, params = tp.utils.count_ops_and_params(model.module, example_inputs)
        print("#Params: {:.4f} M => {:.4f} M".format(base_params/1e6, params/1e6))
        print("#MACS: {:.4f} G => {:.4f} G".format(base_macs/1e9, macs/1e9))
        model.zero_grad()
        del pruner

    #     if args.pruner=='reinit':
    #         def reset_parameters(model):
    #             for m in model.modules():
    #                 if hasattr(m, 'reset_parameters'):
    #                     m.reset_parameters()
    #         reset_parameters(model)

    # pipeline.save_pretrained(args.save_path)
    if args.pruning_ratio>0:
        os.makedirs(os.path.join(args.save_path, "pruned"), exist_ok=True)
        torch.save(model.module, os.path.join(args.save_path, "pruned", f"unet_pruned_{args.pruning_ratio}_{args.thr}.pth"))
