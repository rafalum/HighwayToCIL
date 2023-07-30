import os
import cv2
import numpy as np
import torch

from preprocessing.loading_utils import load_all_from_path, image_to_patches
from preprocessing.preprocessor import preprocess
from utils.utils import np_to_tensor


class ImageDataset(torch.utils.data.Dataset):
    # dataset class that deals with loading the data and making it available by index.

    def __init__(self, path, device, use_patches=True, resize_to=(400, 400)):
        self.path = path
        self.device = device
        self.use_patches = use_patches
        self.resize_to=resize_to
        self.x, self.y, self.n_samples = None, None, None
        self._load_data()
        torch.manual_seed(1337)

    def _load_data(self):  
        # x is for images and y for the label
        self.x = load_all_from_path(os.path.join(self.path, 'images'))[:,:,:,:3] #np array of train/images
        self.y = load_all_from_path(os.path.join(self.path, 'groundtruth')) #np array of train/groundtruth
        
        self.x = np.moveaxis(self.x, -1, 1)

        self.x = [np.moveaxis(img, 0, -1) for img in self.x]
        # self.y = [np.moveaxis(mask, 0, -1) for mask in self.y]

        self.x = np.stack([cv2.resize(img, dsize=self.resize_to) for img in self.x], 0)
        self.y = np.stack([cv2.resize(mask, dsize=self.resize_to) for mask in self.y], 0)
        
        if self.use_patches:  # split each image into patches
            self.x, self.y = image_to_patches(self.x, self.y)
            
        self.x = np.moveaxis(self.x, -1, 1)  # pytorch works with CHW format instead of HWC
        self.n_samples = len(self.x)

    def _preprocess(self, x, y):

        x, y = preprocess(x, y)

        # Convert PIL Images back to tensors
        x = x.contiguous().pin_memory().to(device=self.device, non_blocking=True)
        y = y.contiguous().pin_memory().to(device=self.device, non_blocking=True)

        return x, y

    def __getitem__(self, item):
        return self._preprocess(np_to_tensor(self.x[item], self.device), np_to_tensor(self.y[[item]], self.device))

    def __len__(self):
        return self.n_samples