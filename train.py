import sys
import os

os.environ[
    "CURL_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"  # A workaround in case this happens: https://github.com/mapbox/rasterio/issues/1289
import time
import datetime
import argparse
import copy

import numpy as np
import pandas as pd

from dataloaders.StreamingDatasets import StreamingGeospatialDataset

import torch

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import models
import utils

NUM_WORKERS = 4
NUM_CHIPS_PER_TILE = 100
CHIP_SIZE = 256

parser = argparse.ArgumentParser(description='DFC2021 baseline training script')
parser.add_argument('--input_fn', type=str, required=True,
                    help='The path to a CSV file containing three columns -- "image_fn", "label_fn", and "group" -- that point to tiles of imagery and labels as well as which "group" each tile is in.')
parser.add_argument('--output_dir', type=str, required=True, help='The path to a directory to store model checkpoints.')
parser.add_argument('--overwrite', action="store_true",
                    help='Flag for overwriting `output_dir` if that directory already exists.')
parser.add_argument('--save_most_recent', action="store_true",
                    help='Flag for saving the most recent version of the model during training.')
parser.add_argument('--model', default='unet',
                    choices=(
                        'unet',
                        'fcn',
                        'wakey_fcn',
                        'deeplabv3',
                        'deeplabv3_plus',
                        'fpn',
                        'unet_plus_plus',
                        'manet',
                        'linknet',
                        'pspnet',
                        'pan'
                    ),
                    help='Model to use'
                    )
parser.add_argument('--model_fn', type=str, default='', help='Path to the model file to use.')

## Training arguments
parser.add_argument('--gpu', type=int, default=0, help='The ID of the GPU to use')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size to use for training')
parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train for')
parser.add_argument('--seed', type=int, default=0, help='Random seed to pass to numpy and torch')
args = parser.parse_args()


def image_transforms(img, group):
    if group == 0:
        img = (img - utils.NAIP_2013_MEANS) / utils.NAIP_2013_STDS
    elif group == 1:
        img = (img - utils.NAIP_2017_MEANS) / utils.NAIP_2017_STDS
    else:
        raise ValueError("group not recognized")
    img = np.rollaxis(img, 2, 0).astype(np.float32)
    # img = img[:3, :, :]
    img = torch.from_numpy(img)
    return img


def label_transforms(labels, group):
    # if origin == 1:
    #     labels = utils.NLCD_CLASS_TO_IDX_MAP[labels]  # origin label
    # elif origin == 0:
    #     labels = labels.astype(np.int64)  # fcn_label, landsat_label
    # else:
    #     raise ValueError('origin not recognized')
    labels = labels.astype(np.int64)
    # labels[labels != 1] = 0  # single water label
    labels = torch.from_numpy(labels)
    return labels


def nodata_check(img, labels):
    return np.any(labels == 0) or np.any(np.sum(img == 0, axis=2) == 4)


def main():
    print("Starting DFC2021 baseline training script at %s" % (str(datetime.datetime.now())))

    # -------------------
    # Setup
    # -------------------
    assert os.path.exists(args.input_fn)

    if os.path.isfile(args.output_dir):
        print("A file was passed as `--output_dir`, please pass a directory!")
        return

    if os.path.exists(args.output_dir) and len(os.listdir(args.output_dir)):
        if args.overwrite:
            print(
                "WARNING! The output directory, %s, already exists, we might overwrite data in it!" % (args.output_dir))
        else:
            print(
                "The output directory, %s, already exists and isn't empty. We don't want to overwrite and existing results, exiting..." % (
                    args.output_dir))
            return
    else:
        print("The output directory doesn't exist or is empty.")
        os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda:%d" % args.gpu)
    else:
        print("WARNING! Torch is reporting that CUDA isn't available, exiting...")
        return

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # -------------------
    # Load input data
    # -------------------
    input_dataframe = pd.read_csv(args.input_fn)
    image_fns = input_dataframe["image_fn"].values
    label_fns = input_dataframe["label_fn"].values
    groups = input_dataframe["group"].values
    # origins = input_dataframe['origin'].values


    dataset = StreamingGeospatialDataset(
        imagery_fns=image_fns, label_fns=label_fns, groups=groups, chip_size=CHIP_SIZE,
        num_chips_per_tile=NUM_CHIPS_PER_TILE, windowed_sampling=False, verbose=False,
        image_transform=image_transforms, label_transform=label_transforms, nodata_check=nodata_check
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    num_training_batches_per_epoch = int(len(image_fns) * NUM_CHIPS_PER_TILE / args.batch_size)
    print("We will be training with %d batches per epoch" % (num_training_batches_per_epoch))

    # -------------------
    # Setup training
    # -------------------
    if args.model == "unet":
        model = models.get_unet()
    elif args.model == "fcn":
        model = models.get_fcn()
    elif args.model == 'wakey_fcn':
        model = models.get_wakey_fcn()
    elif args.model == 'deeplabv3':
        model = models.get_deeplabv3()
    elif args.model == 'fpn':
        model = models.get_fpn()
    elif args.model == 'deeplabv3_plus':
        model = models.get_deeplabv3_plus()
    elif args.model == 'unet_plus_plus':
        model = models.get_unet_plus_plus()
    elif args.model == 'manet':
        model = models.get_manet()
    elif args.model == 'linknet':
        model = models.get_linknet()
    elif args.model == 'pspnet':
        model = models.get_pspnet()
    elif args.model == 'pan':
        model = models.get_pan()
    else:
        raise ValueError("Invalid model")

    if args.model_fn:
        model.load_state_dict(torch.load(args.model_fn))
        print('model load checkpoint:{}'.format(args.model_fn))

    # weight0 = torch.from_numpy(np.array(
    #     [5.4848, 3.9617, 5.4848, 3.8687, 4.4298, 4.9357, 5.2594, 5.4078, 3.0675, 4.9381, 3.9661, 5.3283, 5.3877, 3.9875,
    #      2.9660, 3.7543, 4.8434])).float().to(device)
    #
    # weight_clean_fcn_label = torch.from_numpy(np.array(
    #     [5.48481495, 3.52168741, 5.48481495, 4.4037656, 4.80003994, 4.69651623, 5.47566343, 5.48481495, 2.20062309,
    #      5.48257246, 5.30252828, 5.48481495, 5.48481495, 4.14244262, 2.95523514, 4.16211336, 5.08687318]
    # )).float().to(device)
    #
    # weight_val_fcn = torch.from_numpy(np.array(
    #     [5.48481495, 5.00751378, 5.48481495, 0.0005, 0.0005, 3.73850538,
    #      5.48481495, 5.48481495, 2.07523204, 5.48481495, 5.48481495, 5.48481495,
    #      5.48481495, 3.71980016, 3.35288986, 5.45077001, 5.48479066]
    # )).float().to(device)

    # weight_val_fcn_56 = torch.from_numpy(np.array(
    #     [5.48481495, 4.19664424, 5.48481495, 0.0005, 0.0005, 2.75048576,
    #      2.48481495, 7.48481495, 4.93548247, 5.48481495, 5.48481495, 5.48481495,
    #      5.48481495, 4.70433538, 2.98566653, 7.28072808, 7.4841966]
    # )).float().to(device)

    weight_test_fcn_63 = torch.from_numpy(np.array(
        [5.48481495, 4.03948295, 5.48481495, 0.0005, 0.0005, 3.8191025,
         5.48481495, 5.48481495, 1.79941953, 5.48481495, 5.48481495, 5.48481495,
         5.48481495, 3.72170492, 3.26333272, 5.25200058, 5.4848104]
    )).float().to(device)

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, amsgrad=True)
    criterion = nn.CrossEntropyLoss(weight=weight_test_fcn_63, size_average=True)
    # criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    print("Model has %d parameters" % (utils.count_parameters(model)))

    # -------------------
    # Model training
    # -------------------
    training_task_losses = []
    num_times_lr_dropped = 0
    model_checkpoints = []

    for epoch in range(args.num_epochs):
        lr = utils.get_lr(optimizer)

        training_losses = utils.fit(
            model,
            device,
            dataloader,
            num_training_batches_per_epoch,
            optimizer,
            criterion,
            epoch,
        )
        scheduler.step(training_losses[0])

        model_checkpoints.append(copy.deepcopy(model.state_dict()))
        if args.save_most_recent:
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'checkpoint_epoch{}.pt'.format(epoch)))

        if utils.get_lr(optimizer) < lr:
            num_times_lr_dropped += 1
            print("")
            print("Learning rate dropped")
            print("")

        training_task_losses.append(training_losses[0])

        if num_times_lr_dropped == 4:
            break

    # -------------------
    # Save everything
    # -------------------
    save_obj = {
        'args': args,
        'training_task_losses': training_task_losses,
        "checkpoints": model_checkpoints
    }

    save_obj_fn = "results.pt"
    with open(os.path.join(args.output_dir, save_obj_fn), 'wb') as f:
        torch.save(save_obj, f)


if __name__ == "__main__":
    main()
