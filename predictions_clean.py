## 该脚本用来在提交的predictions结果上面滤去杂波

import os
from tqdm import tqdm

os.environ[
    "CURL_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"  # A workaround in case this happens: https://github.com/mapbox/rasterio/issues/1289
import rasterio
import cv2

output_dir = '/root/DFC2021/dfc2021-msd-baseline/results/vote_9_3_8/submission/t_deploy_56ed_process4_ly_imprev_clean_add'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

prediction_names = os.listdir(
    '/root/DFC2021/dfc2021-msd-baseline/results/vote_9_3_8/submission/t_deploy_56ed_process4_ly_imprev')

before_dir = '/root/DFC2021/dfc2021-msd-baseline/results/vote_9_3_8/submission/t_deploy_56ed_process4_ly_imprev_clean_add'

for name in tqdm(prediction_names):
    info = rasterio.open(os.path.join(before_dir, name))
    input_width, input_height = info.width, info.height
    input_profile = info.profile.copy()

    before_path = os.path.join(before_dir, name)
    before_data = rasterio.open(before_path).read(1)

    data = cv2.medianBlur(before_data, 5)
    # pre_area_threshold = 105
    # pre_length_threshold = 65

    pre_area_threshold = 110
    pre_length_threshold = 70

    contours, hierarch = cv2.findContours(data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        length = cv2.arcLength(contours[i], True)
        if area < pre_area_threshold and length < pre_length_threshold:
            cv2.drawContours(data, [contours[i]], 0, (0, 0, 255), -1)

    output_profile = input_profile.copy()
    output_profile["count"] = 1

    output_fn = os.path.join(output_dir, name)

    with rasterio.open(output_fn, "w", **output_profile) as f:
        f.write(data, 1)
