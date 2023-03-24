# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2023-03-04

This script uses Ray Tune to tune hyperparameters for deep steganography. It 
heavily borrows from the Ray Tune tutorial available at 
https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html. 
"""


from functools import partial
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torchvision.utils import save_image
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp.grad_scaler import GradScaler

import torchvision
import torchvision.transforms as transforms

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

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


def train(config):
    """This function is expected to train a model using the given hyperparameters.
    :param config: A dictionary of hyperparameters.
    :param checkpoint_dir: The directory to load checkpoints from.
    :param data_dir: The directory to load data from.
    """
    
    Hnet = HideNet(
        first_c=config["first_channels"],  # USE SAME FOR BOTH TO REDUCE SEARCH SPACE
        max_c=config["max_channels"],
        n_conv=config["n_convs"],
        upsampling_mode=config["upsampling_mode"],
        n_depthwise=config["n_depthwise"]
    )

    Hnet.cuda()
    Hnet.apply(weights_init)
    # if config["checkpoint_dir"]:
    #     Hnet.load_state_dict(
    #         torch.load(os.path.join(config["checkpoint_dir"], "checkpoint"))["Hnet"]
    #     )

    Rnet = RevealNet(
        output_function=nn.Sigmoid,
        nhf=config["first_channels"] # USE SAME FOR BOTH TO REDUCE SEARCH SPACE
    )
    Rnet.cuda()
    Rnet.apply(weights_init)
    # if config["checkpoint_dir"]:
    #     Rnet.load_state_dict(
    #         torch.load(os.path.join(config["checkpoint_dir"], "checkpoint"))["Rnet"]
    #     )

    Hnet_parameters = filter(lambda p: p.requires_grad, Hnet.parameters())
    Hnet_parameters = sum([np.prod(p.size()) for p in Hnet_parameters])

    Rnet_parameters = filter(lambda p: p.requires_grad, Rnet.parameters())
    Rnet_parameters = sum([np.prod(p.size()) for p in Rnet_parameters])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            Rnet = torch.nn.DataParallel(Rnet).cuda()
            Hnet = torch.nn.DataParallel(Hnet).cuda()

    Rnet.to(device)
    Hnet.to(device)

    criterion = nn.HuberLoss(delta=config["delta"]).cuda()
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

    # if config["checkpoint_dir"]:
    #     checkpoint = torch.load(os.path.join(config["checkpoint_dir"], "checkpoint"))
    #     optimizerH.load_state_dict(checkpoint["optimizerH"])
    #     optimizerR.load_state_dict(checkpoint["optimizerR"])
    #     schedulerH.load_state_dict(checkpoint["schedulerH"])
    #     schedulerR.load_state_dict(checkpoint["schedulerR"])

    trainset, _ = load_data(config["data_dir"])

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

    scaler = GradScaler()

    for epoch in range(config["num_epochs"]):  # loop over the dataset multiple times
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

            with torch.autocast("cuda" if "cuda" in device else "cpu"):

                all_pics = data
                this_batch_size = int(all_pics.size()[0] / 2)

                cover_img = all_pics[0:this_batch_size, :, :, :]  # batchsize,3,256,256
                secret_img = all_pics[this_batch_size : this_batch_size * 2, :, :, :]

                concat_img = torch.cat([cover_img, secret_img], dim=1)

                if "cuda" in device:
                    cover_img = cover_img.cuda()
                    secret_img = secret_img.cuda()
                    concat_img = concat_img.cuda()

                concat_imgv = torch.clone(concat_img)
                concat_imgv.requires_grad = True
                cover_imgv = torch.clone(cover_img)
                cover_imgv.requires_grad = True

                container_img = Hnet(concat_imgv)
                errH = criterion(container_img, cover_imgv)
                train_h_losses.append(errH.item())

                rev_secret_img = Rnet(container_img)
                secret_imgv = torch.clone(secret_img)
                secret_imgv.requires_grad = True
                errR = criterion(rev_secret_img, secret_imgv)
                train_r_losses.append(errR.item())

                betaerrR_secret = config["beta"] * errR
                err_sum = errH + betaerrR_secret
                train_sum_losses.append(err_sum.item())

            scaler.scale(err_sum).backward()
            scaler.step(optimizerH)
            scaler.step(optimizerR)

            train_batch_sizes += this_batch_size
            scaler.update()

        # Validate the model
        Hnet.eval()
        Rnet.eval()

        val_h_losses = []
        val_r_losses = []
        val_sum_losses = []
        val_batch_sizes = 0.0
        for i, (data,_) in enumerate(valloader, 0):
            with torch.no_grad():
                with torch.autocast("cuda" if "cuda" in device else "cpu"):
                    all_pics = data
                    this_batch_size = int(all_pics.size()[0] / 2)

                    cover_img = all_pics[0:this_batch_size, :, :, :]  # batchsize,3,256,256
                    secret_img = all_pics[this_batch_size : this_batch_size * 2, :, :, :]

                    concat_img = torch.cat([cover_img, secret_img], dim=1)

                    if "cuda" in device:
                        cover_img = cover_img.cuda()
                        secret_img = secret_img.cuda()
                        concat_img = concat_img.cuda()

                    concat_imgv = torch.clone(concat_img)
                    concat_imgv.requires_grad = True
                    cover_imgv = torch.clone(cover_img)
                    cover_imgv.requires_grad = True

                    container_img = Hnet(concat_imgv)
                    errH = criterion(container_img, cover_imgv)
                    val_h_losses.append(errH.item())

                    rev_secret_img = Rnet(container_img)
                    secret_imgv = torch.clone(secret_img)
                    secret_imgv.requires_grad = True
                    errR = criterion(rev_secret_img, secret_imgv)
                    val_r_losses.append(errR.item())

                    betaerrR_secret = config["beta"] * errR
                    err_sum = errH + betaerrR_secret
                    val_sum_losses.append(err_sum.item())
                    val_batch_sizes += this_batch_size
        
        schedulerH.step(np.sum(val_sum_losses) / val_batch_sizes)
        schedulerR.step(np.sum(val_sum_losses) / val_batch_sizes)

        # save images
        save_image(container_img, os.path.join(config["checkpoint_dir"],"container.png"), normalize=True)
        save_image(cover_imgv, os.path.join(config["checkpoint_dir"],"cover.png"), normalize=True)
        save_image(rev_secret_img, os.path.join(config["checkpoint_dir"],"revealed_secret.png"), normalize=True)
        save_image(secret_imgv, os.path.join(config["checkpoint_dir"],"secret.png"), normalize=True)

        with tune.checkpoint_dir(os.path.join(config["checkpoint_dir"], str(epoch))) as cp_dir:
            path = os.path.join(cp_dir, "checkpoint")
            checkpoint_dict = {
                "optimizerH": optimizerH.state_dict(),
                "optimizerR": optimizerR.state_dict(),
                "schedulerH": schedulerH.state_dict(),
                "schedulerR": schedulerR.state_dict(),
                "Hnet": Hnet.state_dict(),
                "Rnet": Rnet.state_dict(),
            }
            torch.save(checkpoint_dict, path)

        tune.report(
            {
                "h_train_loss": np.sum(train_h_losses) / train_batch_sizes,
                "r_train_loss": np.sum(train_r_losses) / train_batch_sizes,
                "sum_train_loss": np.sum(train_sum_losses) / train_batch_sizes,
                "h_val_loss": np.sum(val_h_losses) / val_batch_sizes,
                "r_val_loss": np.sum(val_r_losses) / val_batch_sizes,
                "sum_val_loss": np.sum(val_sum_losses) / val_batch_sizes,
                "total_params": Hnet_parameters + Rnet_parameters,
                "weighted_val_loss": math.log(Hnet_parameters + Rnet_parameters) * (np.sum(val_sum_losses) / val_batch_sizes)
                
            }
        )

    print("Finished Training")

if __name__ == "__main__":
    # TODO: Make below command line arguments
    MAX_EPOCHS = 3
    GPU_PER_TRIAL = 1
    CPU_PER_TRIAL = 8
    N_SAMPLES = 100  # bump this up
    BATCH_SIZE = 128

    config = {
        "data_dir": os.path.abspath("./data"),
        "checkpoint_dir":os.path.abspath("./checkpoints"),
        "num_epochs": 30,
        "first_channels": tune.choice([16, 32, 64, 128]),  # number of channels in first layer
        "max_channels": tune.choice([128, 256, 512, 1024, 2048, 4096]),  # maximum number of channels in any layer
        "n_convs": tune.randint(1,8),
        "adam_beta": tune.loguniform(0.93, 0.999),
        "beta": 1,  # Default is 0.75
        "lr": tune.loguniform(1e-6, 1e-2),
        "batch_size": tune.choice([16, 32, 64, 128]),  # set to a power of 2; depends on GPU capacity,
        "delta": tune.uniform(0.1, 1),
        "upsampling_mode": tune.choice(['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear']),
        "n_depthwise": tune.randint(1,8),
    }

    # TODO: Put any initial guesses as to good hyperparameter configurations here
    initial_params = [
        {
            "data_dir": os.path.abspath("./data"),
            "checkpoint_dir":os.path.abspath("./checkpoints"),
            "num_epochs": 30,
            "first_channels": 64,  # number of channels in first layer
            "max_channels": 512,  # maximum number of channels in any layer
            "n_convs": 5,
            "adam_beta": 0.95,
            "beta": 1,  # Default is 0.75
            "lr": 1e-3,
            "batch_size": BATCH_SIZE,  # set to a power of 2; depends on GPU capacity
            "delta": 0.5,
            "upsampling_mode": "nearest",
            "n_depthwise": 4,
        }
    ]

    scheduler = ASHAScheduler(max_t=MAX_EPOCHS, grace_period=1, reduction_factor=2)

    # save parameters to a file
    algo = HyperOptSearch(points_to_evaluate=initial_params)
    #if os.path.exists("search_checkpoint.pkl"):
    #    algo.restore("search_checkpoint.pkl")

    tuner = tune.Tuner(
        tune.with_resources(
            train,
            resources={"cpu": CPU_PER_TRIAL, "gpu": GPU_PER_TRIAL},
        ),
        tune_config=tune.TuneConfig(
            metric="_metric/weighted_val_loss",
            mode="min",
            scheduler=scheduler,
            num_samples=N_SAMPLES,
            search_alg=algo,
        ),
        param_space=config,
    )
    results = tuner.fit()

    algo.save("search_checkpoint.pkl")
    results.get_dataframe().to_csv("results.csv")
