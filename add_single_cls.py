## 该脚本用来叠加任何单分类结果
import os
from tqdm import tqdm

os.environ[
    "CURL_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"  # A workaround in case this happens: https://github.com/mapbox/rasterio/issues/1289
import rasterio
import utils

output_dir = '/root/DFC2021/dfc2021-msd-baseline/results/vote_9_3_8/output/t_deploy_56ed_process4_ly_imprev'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

prediction_names = os.listdir(
    '/root/DFC2021/dfc2021-msd-baseline/results/vote_9_3_8/output/t_deploy_56ed_process4_ly_imprev')

before_dir = '/root/DFC2021/dfc2021-msd-baseline/results/vote_9_3_8/output/t_deploy_56ed_process4_ly_imprev'
single_dir = '/root/DFC2021/dfc2021-msd-baseline/results/test_63_fcn_weight/output/t_ep4_56ed'

for name in tqdm(prediction_names):
    info = rasterio.open(os.path.join(before_dir, name))
    input_width, input_height = info.width, info.height
    input_profile = info.profile.copy()

    before_path = os.path.join(before_dir, name)
    before_data = rasterio.open(before_path).read(1)

    single_path = os.path.join(single_dir, name)
    single_data = rasterio.open(single_path).read(1)

    #################### water 1 #####################
    #
    single_indexes = single_data == 5
    before_data[single_indexes] = 5
    #
    #################### water 1 #####################

    ##################### lv 12 13 14 7 #####################

    # single_indexes = single_data == 12
    # before_data[single_indexes == True] = 12
    #
    # single_indexes = single_data == 13
    # before_data[single_indexes == True] = 13
    #
    # single_indexes = single_data == 14
    # before_data[single_indexes == True] = 14
    #
    # single_indexes = single_data == 7
    # before_data[single_indexes == True] = 7
    #
    ##################### lv 12 13 14 7 #####################

    ##################### Tree 8 9 10 11 15 16 #####################
    # single_indexes = single_data == 8
    # before_data[single_indexes == True] = 8
    #
    # single_indexes = single_data == 9
    # before_data[single_indexes == True] = 9
    #
    # single_indexes = single_data == 10
    # before_data[single_indexes == True] = 10
    #
    # single_indexes = single_data == 11
    # before_data[single_indexes == True] = 11
    #
    # single_indexes = single_data == 15
    # before_data[single_indexes == True] = 15
    #
    # single_indexes = single_data == 16
    # before_data[single_indexes == True] = 16

    ##################### Tree 8 9 10 11 15 16 #####################

    output_profile = input_profile.copy()
    output_profile["driver"] = "GTiff"
    output_profile["dtype"] = "uint8"
    output_profile["count"] = 1
    output_profile["nodata"] = 0

    output_fn = os.path.join(output_dir, name)

    with rasterio.open(output_fn, "w", **output_profile) as f:
        f.write(before_data, 1)
        f.write_colormap(1, utils.NLCD_IDX_COLORMAP)
