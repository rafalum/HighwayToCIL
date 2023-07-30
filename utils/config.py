# some constants

PATCH_SIZE = 16  # pixels per side of square patches
CUTOFF = 0.25  # minimum average brightness for a mask patch to be classified as containing road

# VAL_SIZE = 2506  # size of the validation set (number of images)
VAL_SIZE = 10  # size of the validation set (number of images)

train_path = "dataset/kaggle/training/"
val_path = "dataset/kaggle/validation/"
test_path = "dataset/kaggle/test/images/"

checkpoint_path = "checkpoints/"
submission_path = "submissions/"

data_transformations = {
    "rotate" : True,
    "flip": True,
    "color": False
}
