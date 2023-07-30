import torch
from tqdm.notebook import tqdm
from torchvision.transforms import functional as TF
import os
import re
import numpy as np

from preprocessing.preprocessor import *
from preprocessing.loading_utils import *
from utils.config import *

def np_to_tensor(x, device):
    # allocates tensors from np.arrays
    if device == 'cpu':
        return torch.from_numpy(x).cpu()
    else:
        return torch.from_numpy(x).contiguous().pin_memory().to(device=device, non_blocking=True)
    
    
def augment_dataset(path, new_path, resize_to=(400, 400)):  
        
        # x is for images and y for the label
        images = load_all_from_path(os.path.join(path, 'images'))[:,:,:,:3] #np array of train/images
        groundtruths = load_all_from_path(os.path.join(path, 'groundtruth')) #np array of train/groundtruth
        
        images = np.moveaxis(images, -1,1)  # pytorch works with CHW format instead of HWC
        
        images = np_to_tensor(images, "cpu")
        groundtruths = np_to_tensor(groundtruths, "cpu")
        
        images, groundtruths = rotate(images, groundtruths)
        images, groundtruths = flip(images, groundtruths)
        
        os.mkdir(new_path) 
        
        os.mkdir(new_path + 'images/')
        os.mkdir(new_path + 'groundtruth/')
        
        for i, x, y in enumerate(zip(images, groundtruths)):
            x.save(f"{new_path}/images/image_{i}.png")
            y.save(f"{new_path}/groundtruth/groundtruth_{i}.png")
            
            
def create_submission(labels, test_filenames, submission_filename):
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn, patch_array in zip(sorted(test_filenames), labels):
            img_number = int(re.findall(r"\d+", fn)[0])
            for i in range(patch_array.shape[0]):
                for j in range(patch_array.shape[1]):
                    f.write("{:03d}_{}_{},{}\n".format(img_number, j*PATCH_SIZE, i*PATCH_SIZE, int(patch_array[i, j])))

def get_kaggle_prediction(raw_predictions):

    test_pred = np.moveaxis(raw_predictions, 1, -1)  # CHW to HWC
    test_pred = np.stack([cv2.resize(img, dsize=(400, 400)) for img in test_pred], 0)  # resize to original shape
    # now compute labels
    test_pred = test_pred.reshape((-1, 400 // PATCH_SIZE, PATCH_SIZE, 400 // PATCH_SIZE, PATCH_SIZE))
    test_pred = np.moveaxis(test_pred, 2, 3)
    kaggle_predictions = np.round(np.mean(test_pred, (-1, -2)) > CUTOFF)

    return kaggle_predictions


def ensembling(predictions):

    scale_factor = 1 / len(predictions)

    preds_scaled = []
    for prediction in predictions:
        tensor_scaled = []
        for tensor in prediction:
            tensor_scaled.append(scale_factor * tensor)

        preds_scaled.append(tensor_scaled)

    comb = np.add(preds_scaled[0], preds_scaled[1])

    return comb
            
        
        
        
        
        
        
        
        

           