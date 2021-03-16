import numpy as np
import os
from tqdm import tqdm

os.environ[
    "CURL_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"  # A workaround in case this happens: https://github.com/mapbox/rasterio/issues/1289
import rasterio
import utils

output_dir = './results/vote_9_3_8/output/t_deploy'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# model_sub_dirs = ['./results/clean_60_train_deeplabv3plus/output/ep12_56',
#                   './results/clean_60_train_fcn/output/ep2_58',
#                   './results/clean_60_train_fcn/output/ep4_56',
#                   './results/fcn_both_clean_fcnlabel_conv7/output/ep12_57',
#                   './results/val_train_fcn/output/ep2_57',
#                   './results/clean_60_train_fpn/output/ep7_55',
#                   './results/val_60_separate/output/ep4_54',
#                   './results/clean_60_train_unet/output/ep12_55'] # 0.6171


# model_sub_dirs = ['./results/clean_60_train_deeplabv3plus/output/ep12_56',
#                   './results/clean_60_train_fcn/output/ep2_58',
#                   './results/clean_60_train_fcn/output/ep4_56',
#                   './results/fcn_both_clean_fcnlabel_conv7/output/ep12_57',
#                   './results/val_train_fcn/output/ep2_57',
#                   './results/clean_60_train_fpn/output/ep7_55'] # 0.6131

# model_sub_dirs = ['./results/fcn_both_clean_fcnlabel_conv7/output/imprev_ep1_water_ep1',
#                   './results/fcn_both_clean_fcnlabel_conv7/output/water_nir_ep3',
#                   './results/val_train_fcn_ep30/output/ep5',
#                   './results/fcn_both_clean_fcnlabel_conv7/output/lv_ep1_water_ep1']  # 0.5983

# model_sub_dirs = ['./results/clean_60_train_deeplabv3plus/output/ep12_56',
#                   './results/clean_60_train_fcn/output/ep2_58',
#                   './results/clean_60_train_fcn/output/ep4_56',
#                   './results/fcn_both_clean_fcnlabel_conv7/output/ep12_57',
#                   './results/val_train_fcn/output/ep2_57',
#                   './results/clean_60_train_fpn/output/ep7_55',
#                   './results/val_60_separate/output/ep4_54',
#                   './results/clean_60_train_unet/output/ep12_55',
#                   './results/val_train_fcn_ep30/output/ep5',
#                   './results/fcn_both_clean_fcnlabel_conv7/output/water_nir_ep3']

# model_sub_dirs = ['./results/clean_60_train_deeplabv3_18/output/ep7',
#                   './results/clean_60_train_linknet_18/output/ep7',
#                   './results/clean_60_train_pspnet_34/output/ep7',
#                   './results/clean_60_train_pan_34/output/ep7',
#                   './results/clean_60_train_fpn/output/ep7',
#                   './results/clean_60_train_unet/output/ep12_55',
#                   './results/clean_60_train_deeplabv3plus/output/ep12',
#                   './results/fcn_both_clean_fcnlabel_conv7/output/ep12_57',
#                   './results/clean_60_train_unet_plus_plus/output/ep7',
#                   './results/vote_8/output/add_water_ly_up59_4',
#                   './results/clean_60_train_fcn/output/ep2_58',]

# model_sub_dirs = ['./results/clean_60_train_deeplabv3_18/output/ep7',
#                   './results/clean_60_train_linknet_18/output/ep7',
#                   './results/clean_60_train_pspnet_34/output/ep7',
#                   './results/clean_60_train_pan_34/output/ep7',
#                   './results/clean_60_train_fpn/output/ep7',
#                   './results/clean_60_train_unet/output/ep12_55',
#                   './results/clean_60_train_deeplabv3plus/output/ep12',
#                   './results/fcn_both_clean_fcnlabel_conv7/output/ep12_57',
#                   './results/clean_60_train_unet_plus_plus/output/ep7',
#                   './results/vote_8/output/add_water_ly_up59_4',
#                   './results/clean_60_train_fcn/output/ep2_58', ]


# model_sub_dirs = ['/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_deeplabv3plus/output/ep12_after_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/fcn_both_clean_fcnlabel_conv7/output/ep12_after_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_unet_plus_plus/output/ep12_after_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_fcn/output/ep2_after_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_fcn/output/ep4_after_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_unet/output/ep16_after_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_fpn/output/ep7_after_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/val_train_fcn/output/ep2_after_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/val_60_separate/output/ep4_54_56ed', ]  # 0.6219


# model_sub_dirs = ['/root/DFC2021/dfc2021-msd-baseline/results/val_62_train_fcn/output/ep4_after_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/vote_9_3_6/vote_models_8_ed_13_ed_3_ed_add_water_clean',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_pan_34/output/ep7',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/fcn_both_clean_fcnlabel_conv7/output/water_nir_ep1',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_deeplabv3plus/output/ep12_after_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_unet/output/ep16_after_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/fcn_both_clean_fcnlabel_conv7/output/ep12_after_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/val_60_separate/output/ep4_54_56ed']  #


# model_sub_dirs = ['/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_unet_deploy/output/ep10_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_fpn_deploy/output/ep7_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_deeplabv3plus_deploy/output/ep12_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_fcn_deploy/output/ep4_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_linknet_18_deploy/output/ep7_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_pspnet_deploy/output/ep7_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_unet_plus_plus_deploy/output/ep7_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_pan_deploy/output/ep7_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/val_62_train_fcn/output/ep4_after_56ed', ]  #


# model_sub_dirs = ['/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_deeplabv3plus/output/ep12_after_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_pan_34/output/ep7_after_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_unet/output/ep16_after_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/fcn_both_clean_fcnlabel_conv7/output/ep12_after_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/val_60_separate/output/ep4_54_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/val_62_train_fcn/output/ep4_after_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_pspnet_deploy/output/ep7_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_linknet_18_deploy/output/ep7_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_unet_plus_plus_deploy/output/ep7_56ed']  #

# model_sub_dirs = ['/root/DFC2021/dfc2021-msd-baseline/results/val_62_train_fcn/output/ep4_after_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_fpn_deploy/output/ep7_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_unet_deploy/output/ep10_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_deeplabv3_18/output/ep7_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/val_60_separate/output/ep4_54_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/val_62_train_fcn/output/ep4_after_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_linknet_18/output/ep7_after_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_deeplabv3_18/output/ep7_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_pspnet_deploy/output/ep7_56ed']


model_sub_dirs = ['/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_deeplabv3_18_deploy/output/t_ep7',
                  '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_deeplabv3plus_deploy/output/t_ep12',
                  '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_fcn_deploy/output/t_ep4',
                  '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_fpn_deploy/output/t_ep7',
                  '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_linknet_18_deploy/output/t_ep7',
                  '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_pan_deploy/output/t_ep7',
                  '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_pspnet_deploy/output/t_ep7',
                  '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_unet_deploy/output/t_ep10',
                  '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_unet_plus_plus_deploy/output/t_ep7', ]

prediction_names = os.listdir(
    '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_deeplabv3_18_deploy/output/t_ep7')

eye = np.eye(17, dtype=np.uint8)

for name in tqdm(prediction_names):
    info = rasterio.open(
        os.path.join('/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_deeplabv3_18_deploy/output/t_ep7',
                     name))
    input_width, input_height = info.width, info.height
    input_profile = info.profile.copy()
    tmp = np.zeros((input_height, input_width, 17), dtype=np.uint8)
    for model_sub in model_sub_dirs:
        img_path = os.path.join(model_sub, name)
        data = rasterio.open(img_path).read(1)
        one_hot = eye[data]
        tmp += one_hot
    target = tmp.argmax(axis=2).astype(np.uint8)

    output_profile = input_profile.copy()
    output_profile["driver"] = "GTiff"
    output_profile["dtype"] = "uint8"
    output_profile["count"] = 1
    output_profile["nodata"] = 0

    output_fn = os.path.join(output_dir, name)

    with rasterio.open(output_fn, "w", **output_profile) as f:
        f.write(target, 1)
        f.write_colormap(1, utils.NLCD_IDX_COLORMAP)
