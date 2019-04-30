import os
import os.path as osp

import numpy as np
import rasterio
from rasterio.windows import Window

from . import misc
from . import preprocessing


def predict_patches(predictor, patch_configs, pad=16, use_origial_scale=True, dst_dir='./output'):
    os.makedirs(dst_dir, exist_ok=True)
    scene_ids = sorted(set(patch_configs.scene_id))
    for scene_id in scene_ids:
        print('Predicting scene id: {}'.format(scene_id))
        extracted_patch_configs = patch_configs[patch_configs.scene_id==scene_id]
        with rasterio.open(extracted_patch_configs.iloc[0, :]['src_files'][0]) as src:
            profile = src.profile
        profile.update(count=3)

        dst_file = osp.join(dst_dir, scene_id + '.tif')
        with rasterio.open(dst_file, 'w', **profile) as dst:
            for i, row in extracted_patch_configs.iterrows():
                src_files = row['src_files']
                window_config = row['window_config']

                pan_window_config = {
                    'col_off': window_config['col_off'] - pad,
                    'row_off': window_config['row_off'] - pad,
                    'width': window_config['width'] +  2 * pad,
                    'height': window_config['height'] +  2 * pad
                }
                ms_window_config = {
                    'col_off': (window_config['col_off'] - pad) // 2,
                    'row_off': (window_config['row_off'] - pad) // 2,
                    'width': (window_config['width'] +  2 * pad) // 2,
                    'height': (window_config['height'] +  2 * pad) // 2
                }
                pan_arr = misc.read_data(src_files[0], pan_window_config)
                ms_arrs = [misc.read_data(f, ms_window_config) for f in src_files[1:]]
                input = preprocessing.prepare_input(pan_arr, ms_arrs, use_origial_scale)
                output = predictor.predict([input])[0] # Shape of output is (C, H, W)
                output = misc.crop_center(output, width=window_config['width'], height=window_config['height'])

                nodata_mask = misc.get_nodata_mask(src_files, window_config)
                num_ms_bands = len(src_files) - 1

                for i in range(num_ms_bands):
                    dst_arr = output[i, :, :]
                    dst_arr[nodata_mask] = 0
                    dst.write(dst_arr.astype(profile['dtype']), window=Window(**window_config), indexes=i+1)
