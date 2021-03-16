# 随机森林

import re
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import rasterio
from tqdm import tqdm
import os
import torch
from dataloaders.TileMLDatasets import TileMLTrainDataset
import utils

NUM_WORKERS = 4
CHIP_SIZE = 256
PADDING = 128
assert PADDING % 2 == 0
HALF_PADDING = PADDING // 2
CHIP_STRIDE = CHIP_SIZE - PADDING

parser = argparse.ArgumentParser(description='DFC2021 model ML script')
parser.add_argument('--input_fn', type=str, required=True,
                    help='The path to a CSV file containing three columns -- "image_fn", "label_fn", and "group" -- that point to tiles of imagery and labels as well as which "group" each tile is in.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size to use during inference.')
parser.add_argument('--std', action='store_true')

args = parser.parse_args()


def read():
    input_dataframe = pd.read_csv(args.input_fn)
    image_fns = input_dataframe["image_fn"].values
    label_fns = input_dataframe["label_fn"].values
    groups = input_dataframe["group"].values

    total_data = []
    total_label = []

    for image_idx in tqdm(range(len(image_fns))):
        tic = time.time()
        image_fn = image_fns[image_idx]
        image_fn = re.findall(r'\bdata.*\b', image_fn)[0]
        image_fn = image_fn.split('/')
        image_fn = os.path.join('./data/image', image_fn[1], image_fn[2])

        label_fn = label_fns[image_idx]
        label_fn = re.findall(r'\bdata.*\b', label_fn)[0]
        label_fn = label_fn.split('/')
        ################################################ origin label
        label_fn = os.path.join('./data/image', label_fn[1], label_fn[2])
        ################################################ fcn_label
        # label_fn_2 = label_fn[2]
        # if label_fn_2.endswith('2016.tif'):
        #     label_fn_2 = label_fn_2.replace('2016', '2017')
        # label_fn = os.path.join('/root/DFC2021/dfc2021-msd-baseline/data/fcn_label', label_fn[1],
        #                         label_fn_2)

        group = groups[image_idx]

        print("(%d/%d) Processing %s" % (image_idx, len(image_fns), image_fn), end=" ... ")

        def data_image_transforms(img):
            if args.std:
                if group == 0:
                    img = (img - utils.NAIP_2013_MEANS) / utils.NAIP_2013_STDS
                elif group == 1:
                    img = (img - utils.NAIP_2017_MEANS) / utils.NAIP_2017_STDS
                else:
                    raise ValueError("group not recognized")
            img = torch.from_numpy(img)
            return img

        def label_image_transforms(labels):
            labels = utils.NLCD_CLASS_TO_IDX_MAP[labels]  # origin label
            # labels = labels.astype(np.int64)  # fcn_label
            # labels = torch.from_numpy(labels)
            return labels

        with rasterio.open(image_fn) as f:
            input_width, input_height = f.width, f.height
            input_profile = f.profile.copy()

        dataset = TileMLTrainDataset(image_fn, label_fn, chip_size=CHIP_SIZE, stride=CHIP_STRIDE,
                                     image_transform=data_image_transforms, label_transform=label_image_transforms,
                                     verbose=False)
        # print('dataset len: {}'.format(len(dataset)))
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        item_data = []
        item_label = []
        for i, (data, label, coords) in enumerate(dataloader):
            data = data.numpy()  # (32, 256, 256, 4)
            label = label.numpy()  # (32, 256, 256, 1)
            item_data.append(data)
            item_label.append(label)
        item_data = np.concatenate(item_data, 0)  # (600, 256, 256, 4)
        item_label = np.concatenate(item_label, 0)  # (600, 256, 256, 1)
        total_data.append(item_data)
        total_label.append(item_label)

    total_data = np.concatenate(total_data, 0)  # (88980, 256, 256, 4)
    total_label = np.concatenate(total_label, 0)  # (88980, 256, 256, 1)

    total_data = total_data.reshape([-1, 4])
    total_label = total_label.reshape([-1, 1])
    print('total_data shape:{}'.format(total_data.shape))
    print('total_label shape:{}'.format(total_label.shape))
    np.save('/root/DFC2021/dfc2021-msd-baseline/data/npy/1data.npy', total_data)
    np.save('/root/DFC2021/dfc2021-msd-baseline/data/npy/1label.npy', total_label)
    return total_data, total_label


def train(data, label):
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=1)
    print('training......')
    rf.fit(data, label)
    joblib.dump(rf, '/root/DFC2021/dfc2021-msd-baseline/data/npy/rfc.pkl')
    return 0


if __name__ == '__main__':
    data_npy_path = '/root/DFC2021/dfc2021-msd-baseline/data/npy/1data.npy'
    label_npy_path = '/root/DFC2021/dfc2021-msd-baseline/data/npy/1label.npy'

    if os.path.exists(data_npy_path) and os.path.exists(label_npy_path):
        data = np.load(data_npy_path)
        label = np.load(label_npy_path)
        print('total_data shape:{}'.format(data.shape))
        print('total_label shape:{}'.format(label.shape))
    else:
        data, label = read()
        print('total_data shape:{}'.format(data.shape))
        print('total_label shape:{}'.format(label.shape))
    train_start = time.time()
    train(data, label)
    train_end = time.time()
    print('train finished, time:{}'.format(train_end - train_start))
