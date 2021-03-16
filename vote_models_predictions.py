import numpy as np
import os
from tqdm import tqdm

os.environ[
    "CURL_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"  # A workaround in case this happens: https://github.com/mapbox/rasterio/issues/1289
import rasterio

output_dir = '/root/DFC2021/dfc2021-msd-baseline/results/vote_9_3_7_deploy/submission/predictions'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# model_sub_dirs = ['/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_deeplabv3plus/submission/ep12',
#                   './results/clean_60_train_fcn/output/ep2_58',
#                   './results/clean_60_train_fcn/output/ep4_56',
#                   './results/fcn_both_clean_fcnlabel_conv7/output/ep12_57',
#                   './results/val_train_fcn/output/ep2_57',
#                   './results/clean_60_train_fpn/output/ep7_55',
#                   './results/val_60_separate/output/ep4_54',
#                   './results/clean_60_train_unet/output/ep12_55']  # 0.6171

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

# model_sub_dirs = ['/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_deeplabv3plus/submission/ep12_after_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/fcn_both_clean_fcnlabel_conv7/submission/ep12_after_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_unet_plus_plus/submission/ep12_after_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_fcn/submission/ep2_after_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_fcn/submission/ep4_after_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_unet/submission/ep16_after_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_fpn/submission/ep7_after_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/val_train_fcn/submission/ep2_after_56ed',
#                   '/root/DFC2021/dfc2021-msd-baseline/results/val_60_separate/submission/ep4_54_56ed', ]  # 0.6165


model_sub_dirs = ['/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_unet_deploy/submission/ep10_56ed',
                  '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_fpn_deploy/submission/ep7_56ed',
                  '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_deeplabv3plus_deploy/submission/ep12_56ed',
                  '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_fcn_deploy/submission/ep4_56ed',
                  '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_linknet_18_deploy/submission/ep7_56ed',
                  '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_pspnet_deploy/submission/ep7_56ed',
                  '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_unet_plus_plus_deploy/submission/ep7_56ed',
                  '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_pan_deploy/submission/ep7_56ed',
                  '/root/DFC2021/dfc2021-msd-baseline/results/val_62_train_fcn/submission/ep4_after_56ed', ]  #

prediction_names = os.listdir(
    '/root/DFC2021/dfc2021-msd-baseline/results/clean_60_train_unet_deploy/submission/ep10_56ed')

eye = np.eye(16, dtype=np.uint8)

for name in tqdm(prediction_names):
    info = rasterio.open(os.path.join(model_sub_dirs[0], name))
    input_width, input_height = info.width, info.height
    input_profile = info.profile.copy()
    target_img = info.read(1)

    tmp = np.zeros((input_height, input_width, 16), dtype=np.uint8)

    for model_sub in model_sub_dirs:
        img_path = os.path.join(model_sub, name)
        data = rasterio.open(img_path).read(1)
        one_hot = eye[data]
        tmp += one_hot
    target = tmp.argmax(axis=2).astype(np.uint8)

    output_fn = os.path.join(output_dir, name)
    input_profile["count"] = 1
    with rasterio.open(output_fn, "w", **input_profile) as f:
        f.write(target_img, 1)
