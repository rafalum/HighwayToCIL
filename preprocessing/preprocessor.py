import random
import numpy as np
from PIL import ImageEnhance, ImageFilter
from torchvision.transforms import functional as TF

from utils.config import data_transformations

# rotate the images and crop the black corners
def rotate(images, groundtruths):
    
    result_x = []
    result_y = []
    
    angles = [angle for angle in range(0, 25, 10)]

    for x,y in zip(images, groundtruths):
        
        x = TF.to_pil_image(x)
        y = TF.to_pil_image(y)

        # Rotations
        for angle in angles: 

            original_size = x.size
            new_size = tuple(map(lambda i, j: i - j, original_size, (4 * abs(angle), 4 * abs(angle))))

            x_rot = TF.rotate(x, angle)
            x_rot = TF.center_crop(x, new_size)
            x_rot = TF.resize(x, original_size)

            y_rot = TF.rotate(y, angle)
            y_rot = TF.center_crop(y, new_size)
            y_rot = TF.resize(y, original_size)

            result_x.append(x_rot)
            result_y.append(y_rot)

    return result_x, result_y


# creates a vertical, horizontal and 90 deg rotated version of the images
def flip(images, groundtruths):
    
    result_x = images.copy()
    result_y = groundtruths.copy()
    
    for x,y in zip(images, groundtruths): 
        
        # Horizontal Flipping
        x_hflip = TF.hflip(x)
        y_hflip = TF.hflip(y)

        result_x.append(x_hflip)
        result_y.append(y_hflip)

        # Vertical Flipping
        x_vflip = TF.vflip(x)
        y_vflip = TF.vflip(y)

        result_x.append(x_vflip)
        result_y.append(y_vflip)
        
        # Horizontal & Vertical
        x_hvflip = TF.vflip(x_hflip)
        y_hvflip = TF.vflip(y_hflip)

        result_x.append(x_hvflip)
        result_y.append(y_hvflip)
        
    return result_x, result_y


# creates a color transform with a certain probability to the input x
def color_transform(x):
    prob = random.random()

    # 15% edge enhancement
    enhance_probability = 0.15

    # 15% brightness adjustment
    brightness_probability = 0.30

    # 20% Gaussian noise
    filter_probability = 0.5

    if prob < enhance_probability:
        x = x.filter(ImageFilter.EDGE_ENHANCE_MORE)

    elif enhance_probability <= prob <= brightness_probability:
        brightness_factor = random.uniform(0.5, 1)
        enhancer_x = ImageEnhance.Brightness(x)
        x = enhancer_x.enhance(brightness_factor)

    elif prob <= filter_probability:
        x = x.filter(ImageFilter.BLUR)

    return x

# preforms various data augmentations to increase the size of the dataset
def preprocess(x, y):
        
    # Convert tensors to PIL Images
    x = TF.to_pil_image(x)
    y = TF.to_pil_image(y)

    # Apply the same transformations to x and y
    if data_transformations["rotate"]:
        angle = np.random.normal(0, 10, None)

        x = TF.rotate(x, angle)
        y = TF.rotate(y, angle)

        original_size = x.size
        new_size = tuple(map(lambda i, j: i - j, original_size, (4 * abs(angle), 4 * abs(angle))))

        x = TF.center_crop(x, new_size)
        y = TF.center_crop(y, new_size)

        x = TF.resize(x, original_size)
        y = TF.resize(y, original_size)

    if data_transformations["flip"]:
        should_flip_horizontal = random.random() > 0.5
        should_flip_vertical = random.random() > 0.5

        if should_flip_horizontal:
            x = TF.hflip(x)
            y = TF.hflip(y)
        if should_flip_vertical:
            x = TF.vflip(x)
            y = TF.vflip(y)

    if data_transformations["color"]:
        x = color_transform(x)

    x = TF.to_tensor(x)
    y = TF.to_tensor(y)

    return x, y