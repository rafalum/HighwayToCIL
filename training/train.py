import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm

import math
import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from random import sample
from PIL import Image, ImageEnhance, ImageFilter
import time

# New imports
from torchvision.transforms import functional as TF
import random

from .training_utils import *


from utils.config import * 


def train(train_dataloader, eval_dataloader, model, loss_fn, metric_fns, optimizer, lr_scheduler, n_epochs, model_name):
    # training loop
    logdir = './tensorboard/net'
    writer = SummaryWriter(logdir)  # tensorboard writer (can also log images)

    history = {}  # collects metrics at the end of each epoch

    # early stopping initialization
    patience = 30
    best_val_patch_acc = None
    best_epoch = None
    early_stop_counter = 0

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        # initialize metric list
        metrics = {'loss': [], 'val_loss': []}
        for k, _ in metric_fns.items():
            metrics[k] = []
            metrics['val_'+k] = []

        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{n_epochs}')
        # training
        model.train()
        for (x, y) in pbar:
            optimizer.zero_grad()  # zero out gradients
            y_hat = model(x)  # forward pass
            loss = loss_fn(y_hat, y)
            loss.backward()  # backward pass
            optimizer.step()  # optimize weights

            # log partial metrics
            metrics['loss'].append(loss.item())
            for k, fn in metric_fns.items():
                metrics[k].append(fn(y_hat, y).item())
            pbar.set_postfix({k: sum(v)/len(v) for k, v in metrics.items() if len(v) > 0})

        # validation
        model.eval()
        with torch.no_grad():  # do not keep track of gradients
            for (x, y) in eval_dataloader:
                y_hat = model(x)  # forward pass
                loss = loss_fn(y_hat, y)
                
                # log partial metrics
                metrics['val_loss'].append(loss.item())
                for k, fn in metric_fns.items():
                    metrics['val_'+k].append(fn(y_hat, y).item())
                #print(metrics)

        # summarize metrics, log to tensorboard and display
        epoch_loss = sum(metrics['val_loss']) / len(metrics['val_loss'])
        val_patch_acc = sum(metrics["val_patch_acc"]) / len(metrics["val_patch_acc"])

        history[epoch] = {k: sum(v) / len(v) for k, v in metrics.items()}
        for k, v in history[epoch].items():
          writer.add_scalar(k, v, epoch)
        if epoch % 20 == 0:
            print(' '.join(['\t- '+str(k)+' = '+str(v)+'\n ' for (k, v) in history[epoch].items()]))
            show_val_samples(x.detach().cpu().numpy(), y.detach().cpu().numpy(), y_hat.detach().cpu().numpy())

        # save model checkpoint if it has better loss
        if best_val_patch_acc is None or val_patch_acc > best_val_patch_acc:
            now = time.strftime("%H:%M-%d-%m-%Y")
            torch.save(model.state_dict(), f"{checkpoint_path}checkpoint_{model_name}_{n_epochs}ep_{now}.pth")
            torch.save(model.state_dict(), f"{checkpoint_path}best_checkpoint.pth")
            best_val_patch_acc = val_patch_acc
            best_epoch = epoch
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # early stopping
        if early_stop_counter >= patience:
            print(f'Early stopping at epoch {epoch}, the lowest validation patch accuracy achieved at epoch {best_epoch}')
            print(f'Best validation patch accuracy: {best_val_patch_acc}')
            break
            
        lr_scheduler.step(int(np.mean(val_patch_acc)))


    print('Finished Training')
    # plot loss curves
    plt.plot([v['loss'] for k, v in history.items()], label='Training Loss')
    plt.plot([v['val_loss'] for k, v in history.items()], label='Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()