import cv2
import torch
import argparse
import numpy as np
from glob import glob

from models.Unet import UNet
from models.ResUnet import ResUNet
from models.LinkNet import LinkNet34
from models.DLinknet import DLinkNet34
from models.DLinknet import DLinkNet101
from models.NL_DLinknet import NL_DLinkNet34
from models.DLinknet_JPU import DLinkNet101_JPU

from training.train import train
from training.loss import FocalLoss
from training.loss import HybridLoss
from training.metrics import accuracy_fn, patch_accuracy_fn

from preprocessing.dataloader import ImageDataset
from preprocessing.loading_utils import load_all_from_path

from utils.utils import np_to_tensor, create_submission, get_kaggle_prediction, ensembling


def predict(model, checkpoint, data_path, device):
    model.load_state_dict(torch.load(checkpoint))

    model.eval()

    images = load_all_from_path(data_path)

    size = images.shape[1:3]

    images = np.stack([cv2.resize(img, dsize=(384, 384)) for img in images], 0)[:, :, :, :3]
    images = np_to_tensor(np.moveaxis(images, -1, 1), device)

    raw_predictions = np.concatenate([model(t).detach().cpu().numpy() for t in images.unsqueeze(1)], 0)

    kaggle_predictions = get_kaggle_prediction(raw_predictions)

    return raw_predictions, kaggle_predictions

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='action')

    train_parser = subparsers.add_parser('train')
    predict_parser = subparsers.add_parser('predict')

    for subparser in [train_parser, predict_parser]:
        subparser.add_argument(
            "model",
            type=str,
            nargs='+',
            help="Specify one or more models.",
            choices=[
                "Unet",
                "ResUnet",
                "LinkNet34",
                "DLinkNet34",
                "DLinkNet101",
                "NL_DLinkNet34",
                "DLinkNet101_JPU"
            ]
        )
        
        subparser.add_argument(
            "--data_path",
            type=str,
            help="Specify the data path.",
            required=True
        )

    train_parser.add_argument(
        "--loss",
        type=str,
        help="Specify which loss to train.",
        choices=[
            "BCE",
            "FocalLoss",
            "HybridLoss"
        ]
    )

    train_parser.add_argument(
        "--batch_size",
        type=int,
        help="Specify the batch size to train.",
        default=8
    )

    train_parser.add_argument(
        "--num_epochs",
        type=int,
        help="Specify the maximum number of epochs to train.",
        default=100
    )

    predict_parser.add_argument(
        "--ensemble",
        action='store_true',
        default=False,
        help="Specify if ensembling should be applied.",
    )

    predict_parser.add_argument(
        "--checkpoint",
        type=str,
        nargs='+',
        help="Specify the checkpoint to predict.",
        required=True
    )

    args = parser.parse_args()

    if args.action == 'predict' and args.ensemble and len(args.model) == 1:
        parser.error("--ensemble requires multiple models")

    if args.action == 'predict' and len(args.checkpoint) != len(args.model):
        parser.error("missmatch between checkpoints and models provided")

    if args.action == "train" and len(args.model) > 1:
        parser.error("please only provide one model to train on")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    models = []
    for model in args.model:
        if "Unet" == model:
            models.append(UNet().to(device))
        elif "ResUnet" == model:
            models.append(ResUNet().to(device))
        elif "LinkNet34" == model:
            models.append(LinkNet34().to(device))
        elif "DLinkNet34" == model:
            models.append(DLinkNet34().to(device))
        elif "DLinkNet101" == model:
            models.append(DLinkNet101().to(device))
        elif "NL_DLinkNet34" == model:
            models.append(NL_DLinkNet34().to(device))
        elif "DLinkNet101_JPU" == model:
            models.append(DLinkNet101_JPU().to(device))

    if args.action == "predict":

        model_names = ""
        raw_predictions = []
        for i, model in enumerate(models):
            raw_pred, kaggle_pred = predict(model, args.checkpoint[i], args.data_path, device)
            create_submission(kaggle_pred, glob(args.data_path + '/*.png'), "submission_" + type(model).__name__ + ".csv")

            if args.ensemble:
                model_names += type(model).__name__
                raw_predictions.append(raw_pred)
        
        if args.ensemble:
            raw_ensemble_pred = ensembling(raw_predictions)
            kaggle_ensemble_pred = get_kaggle_prediction(raw_ensemble_pred)

            create_submission(kaggle_ensemble_pred, glob(args.data_path + '/*.png'), "ensemble_submission_" + model_names + ".csv")

    else:

        model = models[0]
        
        optimizer = torch.optim.Adam(model.parameters())
        metric_fns = {'acc': accuracy_fn, 'patch_acc': patch_accuracy_fn}
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, verbose=True)

        train_dataset = ImageDataset(args.data_path + "/training/", device, use_patches=False, resize_to=(384, 384))
        val_dataset = ImageDataset(args.data_path + "/validation/", device, use_patches=False, resize_to=(384, 384))
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

        if args.loss == "BCE":
            loss = torch.nn.BCELoss()
        elif args.loss == "FocalLoss":
            loss = FocalLoss()
        else:
            loss = HybridLoss()

        train(train_dataloader, val_dataloader, model, loss, metric_fns, optimizer, lr_scheduler, args.num_epochs, type(model).__name__)

if __name__ == "__main__":
    main()

    
        

        




