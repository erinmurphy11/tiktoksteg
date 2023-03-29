# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2023-03-04

This script uses Ray Tune to tune hyperparameters for deep steganography. It 
heavily borrows from the Ray Tune tutorial available at 
https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html.
https://github.com/huggingface/accelerate/blob/main/examples/complete_cv_example.py 
"""
from accelerate import Accelerator

import os
import math
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torchvision
import torchvision.transforms as transforms

from model import HideNet, RevealNet, weights_init

def load_data(data_dir="./data"):
    """This function is expected to load data from the given directory, and
    return a train and test set. The data should be in the form of a PyTorch
    Dataset object.
    :param data_dir: The directory to load data from.
    :return: A tuple of (train_set, test_set).

    May have to use file lock to prevent multiple processes from downloading
    """
    transform = transforms.Compose(
        # add transform resize
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    trainset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)

    return trainset, None


def train(config, args):
    """This function is expected to train a model using the given hyperparameters.
    :param config: A dictionary of hyperparameters.
    :param args: Command-line parsed arguments.
    """
    accelerator = Accelerator(
        cpu=args.cpu, 
        mixed_precision=args.mixed_precision, 
        log_with="tensorboard", 
        logging_dir=args.logging_dir
    )

    Hnet = HideNet(
        first_c=config["first_channels"],  # USE SAME FOR BOTH TO REDUCE SEARCH SPACE
        max_c=config["max_channels"],
        n_conv=config["n_convs"],
        upsampling_mode=config["upsampling_mode"],
        n_depthwise=config["n_depthwise"]
    )

    Hnet.apply(weights_init)

    Rnet = RevealNet(
        output_function=nn.Sigmoid,
        nhf=config["first_channels"] # USE SAME FOR BOTH TO REDUCE SEARCH SPACE
    )
    Rnet.apply(weights_init)

    Hnet_parameters = filter(lambda p: p.requires_grad, Hnet.parameters())
    Hnet_parameters = sum([np.prod(p.size()) for p in Hnet_parameters])

    Rnet_parameters = filter(lambda p: p.requires_grad, Rnet.parameters())
    Rnet_parameters = sum([np.prod(p.size()) for p in Rnet_parameters])

    optimizerH = optim.Adam(
        Hnet.parameters(), lr=config["lr"], betas=(config["adam_beta"], 0.999)
    )
    schedulerH = ReduceLROnPlateau(
        optimizerH, mode="min", factor=0.2, patience=5, verbose=True
    )

    optimizerR = optim.Adam(
        Rnet.parameters(), lr=config["lr"], betas=(config["adam_beta"], 0.999)
    )
    schedulerR = ReduceLROnPlateau(
        optimizerR, mode="min", factor=0.2, patience=8, verbose=True
    )

    trainset, _ = load_data(args.data_dir)

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs]
    )

    trainloader = torch.utils.data.DataLoader(
        train_subset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8
    )
    valloader = torch.utils.data.DataLoader(
        val_subset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8
    )

    Hnet, Rnet, optimizerH, optimizerR, schedulerH, schedulerR, trainloader, valloader = accelerator.prepare(
        Hnet, Rnet, optimizerH, optimizerR, schedulerH, schedulerR, trainloader, valloader
    )

    # We need to initialize the trackers we use, and also store our configuration
    if args.with_tracking:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run, config)

    start_epoch = 0 

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        start_epoch = int(training_difference.replace("epoch_", "")) + 1
        resume_step = None


    for epoch in range(start_epoch, config["num_epochs"]):  # loop over the dataset multiple times
        train_h_losses = []
        train_r_losses = []
        train_sum_losses = []
        train_batch_sizes = 0.0
        Hnet.train()
        Rnet.train()

        # Train the model
        for i, (data,_) in enumerate(trainloader, 0):
            Hnet.zero_grad()
            Rnet.zero_grad()

            all_pics = data
            this_batch_size = int(all_pics.size()[0] / 2)

            cover_img = all_pics[0:this_batch_size, :, :, :]  # batchsize,3,256,256
            secret_img = all_pics[this_batch_size : this_batch_size * 2, :, :, :]

            concat_img = torch.cat([cover_img, secret_img], dim=1)

            concat_imgv = torch.clone(concat_img)
            concat_imgv.requires_grad = True
            cover_imgv = torch.clone(cover_img)
            cover_imgv.requires_grad = True

            container_img = Hnet(concat_imgv)
            errH = F.l1_loss(container_img, cover_imgv)
            train_h_losses.append(errH.item())

            rev_secret_img = Rnet(container_img)
            secret_imgv = torch.clone(secret_img)
            secret_imgv.requires_grad = True
            errR = F.l1_loss(rev_secret_img, secret_imgv)
            train_r_losses.append(errR.item())

            betaerrR_secret = config["beta"] * errR
            err_sum = errH + betaerrR_secret
            train_sum_losses.append(err_sum.item())

            accelerator.backward(err_sum)
            optimizerH.step()
            optimizerR.step()
            optimizerH.zero_grad()
            optimizerR.zero_grad()

            train_batch_sizes += this_batch_size

        # Validate the model
        Hnet.eval()
        Rnet.eval()

        val_h_losses = []
        val_r_losses = []
        val_sum_losses = []
        val_batch_sizes = 0.0
        for i, (data,_) in enumerate(valloader, 0):
            with torch.no_grad():
                all_pics = data
                this_batch_size = int(all_pics.size()[0] / 2)

                cover_img = all_pics[0:this_batch_size, :, :, :]  # batchsize,3,256,256
                secret_img = all_pics[this_batch_size : this_batch_size * 2, :, :, :]

                concat_img = torch.cat([cover_img, secret_img], dim=1)

                concat_imgv = torch.clone(concat_img)
                concat_imgv.requires_grad = True
                cover_imgv = torch.clone(cover_img)
                cover_imgv.requires_grad = True

                container_img = Hnet(concat_imgv)
                errH = F.l1_loss(container_img, cover_imgv)
                val_h_losses.append(errH.item())

                rev_secret_img = Rnet(container_img)
                secret_imgv = torch.clone(secret_img)
                secret_imgv.requires_grad = True
                errR = F.l1_loss(rev_secret_img, secret_imgv)
                val_r_losses.append(errR.item())

                betaerrR_secret = config["beta"] * errR
                err_sum = errH + betaerrR_secret
                val_sum_losses.append(err_sum.item())
                val_batch_sizes += this_batch_size
        
        schedulerH.step(np.sum(val_sum_losses) / val_batch_sizes)
        schedulerR.step(np.sum(val_sum_losses) / val_batch_sizes)

        # save imagesdata_dir
        # THIS IS GONNA GIVE ERRORS
        accelerator.get_tracker("tensorboard").tracker.add_image(
            "container", 
            make_grid(container_img, normalize=True),
            epoch
        )
        accelerator.get_tracker("tensorboard").tracker.add_image(
            "cover", 
            make_grid(cover_img, normalize=True),
            epoch
        )
        accelerator.get_tracker("tensorboard").tracker.add_image(
            "revealed_secret", 
            make_grid(rev_secret_img, normalize=True),
            epoch
        )
        accelerator.get_tracker("tensorboard").tracker.add_image(
            "secret", 
            make_grid(secret_imgv, normalize=True),
            epoch
        )
        
        accelerator.print(f"epoch {epoch}")
        if args.with_tracking:
            accelerator.log(
                {
                    "h_train_loss": np.sum(train_h_losses) / train_batch_sizes,
                    "r_train_loss": np.sum(train_r_losses) / train_batch_sizes,
                    "sum_train_loss": np.sum(train_sum_losses) / train_batch_sizes,
                    "h_val_loss": np.sum(val_h_losses) / val_batch_sizes,
                    "r_val_loss": np.sum(val_r_losses) / val_batch_sizes,
                    "sum_val_loss": np.sum(val_sum_losses) / val_batch_sizes,
                    "epoch": epoch,
                },
                step=epoch,
            )

        output_dir = f"epoch_{epoch}"
        if args.output_dir is not None:
            output_dir = os.path.join(args.output_dir, output_dir)
        accelerator.save_state(output_dir)

    accelerator.print("Finished Training")
    if args.with_tracking:
        accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training arguments.")
    parser.add_argument(
        "--data_dir", 
        type= str, 
        default=os.path.abspath("./data"), 
        help="The data folder on disk."
    )
    parser.add_argument("--fp16", action="store_true", help="If passed, will use FP16 training.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Optional save directory where all checkpoint folders will be stored. Default is the current working directory.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        default=True,
        help="Whether to load in all tensorboard experiment trackers from the environment and use them for logging.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Location on where to store experiment tracking logs`",
    )
    args = parser.parse_args()

    config = {
            "num_epochs": 1000,
            "first_channels": 16,  # number of channels in first layer
            "max_channels": 256,  # maximum number of channels in any layer
            "n_convs": 4,
            "adam_beta": 0.97,
            "beta": 1,  # Default is 0.75
            "lr": 3e-4,
            "batch_size": 128,  # set to a power of 2; depends on GPU capacity
            "upsampling_mode": "nearest",
            "n_depthwise": 4,
        }
    train(config, args)