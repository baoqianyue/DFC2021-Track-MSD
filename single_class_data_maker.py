import numpy as np
import rasterio
import utils
import os
from collections import Counter
from tqdm import tqdm
import pandas as pd
import utils
import re

input_fn = './data/splits/training_set_naip_nlcd_2017.csv'
input_dataframe = pd.read_csv(input_fn)
image_fns = input_dataframe["image_fn"].values
label_fns = input_dataframe["label_fn"].values

need_drop_data = []
item = 0
for image_idx in tqdm(range(len(label_fns))):

    label_fn = label_fns[image_idx]
    label_fn = re.findall(r'\bdata.*\b', label_fn)[0]
    label_fn = label_fn.split('/')
    ################################################ origin label
    label_fn = os.path.join('./data/image', label_fn[1], label_fn[2])
    with rasterio.open(label_fn) as lf:
        label_img = lf.read()
        label_img = utils.NLCD_CLASS_TO_IDX_MAP[label_img].flatten()
        cnt = Counter(label_img)
        water_per = cnt[5] / label_img.shape[0]
        print('Medium Intensity per: {}'.format(water_per))
        # 0.001
        if water_per < 0.001:
            item = item + 1
            need_drop_data.append(image_idx)

print('total label num: {}'.format(len(label_fns)))
print('len of need_drop_data: {}'.format(len(need_drop_data)))
print(item / len(label_fns))


df = pd.DataFrame(input_dataframe)
df = df.drop(labels=need_drop_data)
df.to_csv("./data/splits/training_set_naip_nlcd_2013_5.csv", index=False)
