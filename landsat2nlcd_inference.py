import re
import os

os.environ[
    "CURL_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"  # A workaround in case this happens: https://github.com/mapbox/rasterio/issues/1289
import time
import datetime
import argparse

import numpy as np
import pandas as pd

import rasterio

import torch
import torch.nn.functional as F

import models
from dataloaders.Landsat2NlcdTileDatasets import Landsat2NlcdTileDataset
import utils

NUM_WORKERS = 4
CHIP_SIZE = 256
PADDING = 128
assert PADDING % 2 == 0
HALF_PADDING = PADDING // 2
CHIP_STRIDE = CHIP_SIZE - PADDING

parser = argparse.ArgumentParser(description='DFC2021 model inference script')
parser.add_argument('--input_fn', type=str, required=True,
                    help='The path to a CSV file containing three columns -- "image_fn", "label_fn", and "group" -- that point to tiles of imagery and labels as well as which "group" each tile is in.')
parser.add_argument('--model_fn', type=str, required=True, help='Path to the model file to use.')
parser.add_argument('--output_dir', type=str, required=True,
                    help='The path to output the model predictions as a GeoTIFF. Will fail if this file already exists.')
parser.add_argument('--overwrite', action="store_true",
                    help='Flag for overwriting `--output_dir` if that directory already exists.')
parser.add_argument('--gpu', type=int, default=0, help='The ID of the GPU to use')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size to use during inference.')
parser.add_argument('--save_soft', action="store_true",
                    help='Flag that enables saving the predicted per class probabilities in addition to the "hard" class predictions.')
parser.add_argument('--model', default='unet',
                    choices=(
                        'unet',
                        'fcn'
                    ),
                    help='Model to use'
                    )
parser.add_argument('--replace', default='')

args = parser.parse_args()

LANDSAT_2016_MEANS = np.array([1602.33, 2357.00, 360.42])
LANDSAT_2016_STDS = np.array([790.62, 1104.47, 157.94])


def main():
    print("Starting DFC2021 model inference script at %s" % (str(datetime.datetime.now())))

    # -------------------
    # Setup
    # -------------------
    assert os.path.exists(args.input_fn)
    assert os.path.exists(args.model_fn)

    if os.path.isfile(args.output_dir):
        print("A file was passed as `--output_dir`, please pass a directory!")
        return

    if os.path.exists(args.output_dir) and len(os.listdir(args.output_dir)) > 0:
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

    # -------------------
    # Load model
    # -------------------
    if args.model == "unet":
        model = models.get_unet()
    elif args.model == "fcn":
        model = models.get_fcn()
    else:
        raise ValueError("Invalid model")
    model.load_state_dict(torch.load(args.model_fn))
    model = model.to(device)

    # -------------------
    # Run on each line in the input
    # -------------------
    input_dataframe = pd.read_csv(args.input_fn)
    image_fns = input_dataframe["image_fn"].values
    groups = input_dataframe["group"].values
    # print('image_fns : {}'.format(image_fns))

    for image_idx in range(len(image_fns)):
        tic = time.time()
        image_fn = image_fns[image_idx]
        image_fn = re.findall(r'\bdata.*\b', image_fn)[0]
        image_fn = image_fn.split('/')
        img_fn1 = image_fn[1]
        img_fn2 = image_fn[2]
        img_fn1 = img_fn1.replace('naip', 'landsat')
        img_fn2 = img_fn2.replace('naip', 'landsat')
        # print('img_fn1: {}'.format(img_fn1))
        image_fn = os.path.join('./data/image', img_fn1, img_fn2)
        # print('image_fn: {}'.format(image_fn))
        group = groups[image_idx]

        print("(%d/%d) Processing %s" % (image_idx, len(image_fns), image_fn), end=" ... ")

        # -------------------
        # Load input and create dataloader
        # -------------------
        def image_transforms(img):
            img = (img - LANDSAT_2016_MEANS) / LANDSAT_2016_STDS
            img = np.rollaxis(img, 2, 0).astype(np.float32)
            img = torch.from_numpy(img)
            return img

        with rasterio.open(image_fn) as f:
            input_width, input_height = f.width, f.height
            input_profile = f.profile.copy()

        dataset = Landsat2NlcdTileDataset(image_fn, chip_size=CHIP_SIZE, stride=CHIP_STRIDE, transform=image_transforms,
                                          verbose=False)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        # -------------------
        # Run model and organize output
        # -------------------

        output = np.zeros((len(utils.NLCD_CLASSES), input_height, input_width), dtype=np.float32)
        kernel = np.ones((CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
        kernel[HALF_PADDING:-HALF_PADDING, HALF_PADDING:-HALF_PADDING] = 5
        counts = np.zeros((input_height, input_width), dtype=np.float32)

        for i, (data, coords) in enumerate(dataloader):
            data = data.to(device)
            with torch.no_grad():
                t_output = model(data)
                t_output = F.softmax(t_output, dim=1).cpu().numpy()

            for j in range(t_output.shape[0]):
                y, x = coords[j]
                output[:, y:y + CHIP_SIZE, x:x + CHIP_SIZE] += t_output[j] * kernel
                counts[y:y + CHIP_SIZE, x:x + CHIP_SIZE] += kernel

        output = output / counts
        output_hard = output.argmax(axis=0).astype(np.uint8)

        # -------------------
        # Save output
        # -------------------
        output_profile = input_profile.copy()
        output_profile["driver"] = "GTiff"
        output_profile["dtype"] = "uint8"
        output_profile["count"] = 1
        output_profile["nodata"] = 0

        output_fn = image_fn.split("/")[-1]  # something like "546_naip-2013.tif"
        output_fn = output_fn.replace("landsat", "nlcd")
        output_fn = os.path.join(args.output_dir, output_fn)

        with rasterio.open(output_fn, "w", **output_profile) as f:
            f.write(output_hard, 1)
            f.write_colormap(1, utils.NLCD_IDX_COLORMAP)

        if args.save_soft:
            output = output / output.sum(axis=0, keepdims=True)
            output = (output * 255).astype(np.uint8)

            output_profile = input_profile.copy()
            output_profile["driver"] = "GTiff"
            output_profile["dtype"] = "uint8"
            output_profile["count"] = len(utils.NLCD_CLASSES)
            del output_profile["nodata"]

            output_fn = image_fn.split("/")[-1]  # something like "546_naip-2013.tif"
            output_fn = output_fn.replace("naip", "predictions-soft")
            output_fn = os.path.join(args.output_dir, output_fn)

            with rasterio.open(output_fn, "w", **output_profile) as f:
                f.write(output)

        print("finished in %0.4f seconds" % (time.time() - tic))


if __name__ == "__main__":
    main()
