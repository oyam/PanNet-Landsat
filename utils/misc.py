from glob import glob
import os
import os.path as osp
import requests

import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window


def download(url, file_name):
    if not osp.exists(file_name):
        with open(file_name, 'wb') as file:
            print('Download from {} to {}'.format(url, file_name))
            response = requests.get(url)
            if response.status_code != 200:
                raise Exception("Can't get " + url)
            file.write(response.content)

        
def download_landsat8_data(data_dir, baseurl):
    scene_id = osp.basename(baseurl)
    os.makedirs(osp.join(data_dir, scene_id), exist_ok=True)
    
    filenames = [osp.join(data_dir, scene_id, scene_id + '_B{}.TIF'.format(band))
                 for band in (2, 3, 4, 8)]

    urls = [osp.join(baseurl, osp.basename(fname)) for fname in filenames]
    
    for url, fname in zip(urls, filenames):
        download(url, fname)
    
    mtl_fname = osp.join(data_dir, scene_id, scene_id + '_MTL.json')
    mtl_url = osp.join(baseurl ,osp.basename(mtl_fname))
    download(mtl_url, mtl_fname)
    
    return filenames, mtl_fname

def get_patch_configs(src_dir, training_data_rate=0.9, validation_data_rate=0.1, test_data_rate=0.0,
                      patch_width=64, patch_height=64, num_patches=None, random_seed=1, include_partial_nodata=False):
    # Rebalance rate for split for the case that summation of training_data_rate, validation_data_rate, and test_data_rate is not 1.
    summ = training_data_rate + validation_data_rate + test_data_rate
    training_data_rate = training_data_rate / summ
    validation_data_rate = validation_data_rate / summ
    test_data_rate = test_data_rate / summ

    patch_configs = pd.DataFrame(columns=['scene_id', 'src_files', 'metadata_file', 'window_config'])
    for scene_id in os.listdir(src_dir):
        src_files = [
            glob(osp.join(src_dir, scene_id, '*B8.TIF'))[0], # Panchromatic
            glob(osp.join(src_dir, scene_id, '*B2.TIF'))[0], # Blue
            glob(osp.join(src_dir, scene_id, '*B3.TIF'))[0], # Green
            glob(osp.join(src_dir, scene_id, '*B4.TIF'))[0] # Red
        ]
        metadata_file = glob(osp.join(src_dir, scene_id, '*MTL.json')),
        pan_width = patch_width * 2
        pan_height = patch_height * 2
        window_configs = get_window_configs(src_files, pan_width, pan_height, include_partial_nodata)
        for window_config in window_configs:
            patch_config = pd.Series([scene_id, src_files, metadata_file, window_config], index=patch_configs.columns)
            patch_configs = patch_configs.append(patch_config, ignore_index=True)

    if num_patches is not None:
        patch_configs = patch_configs.sample(num_patches, random_state=random_seed)
        patch_configs.reset_index(drop=True, inplace=True)
    patch_indices = np.arange(len(patch_configs))
    num_training_patches = int(len(patch_configs) * training_data_rate)
    num_validation_patches = int(len(patch_configs) * validation_data_rate)
    training_patch_indices = np.random.choice(patch_indices, num_training_patches, replace=False)
    validation_and_test_patch_indices = np.setdiff1d(patch_indices, training_patch_indices)
    validation_patch_indices = np.random.choice(validation_and_test_patch_indices, num_validation_patches, replace=False)
    test_patch_indices = np.setdiff1d(validation_and_test_patch_indices, validation_patch_indices)

    training_patch_configs = patch_configs.loc[training_patch_indices, :]
    validation_patch_configs = patch_configs.loc[validation_patch_indices, :]
    test_patch_configs = patch_configs.loc[test_patch_indices, :]

    if test_data_rate == 0 and len(test_patch_configs) > 0:
        if validation_data_rate != 0:
            validation_patch_configs = pd.concat([validation_patch_configs, test_patch_configs], axis=0)
        else:
            training_patch_configs = pd.concat([training_patch_configs, test_patch_configs], axis=0)
        test_patch_configs = pd.DataFrame(columns=['scene_id', 'src_files', 'metadata_file', 'window_config'])

    training_patch_configs.reset_index(drop=True, inplace=True)
    validation_patch_configs.reset_index(drop=True, inplace=True)
    test_patch_configs.reset_index(drop=True, inplace=True)

    return training_patch_configs, validation_patch_configs, test_patch_configs


def get_window_configs(src_files, pan_window_width, pan_window_height, include_partial_nodata=False):
    window_configs = []
    with rasterio.open(src_files[0]) as src:
        global_width = src.width
        global_height = src.height

    for row_offset in range(0, global_height, pan_window_height):
        for col_offset in range(0, global_width, pan_window_width):
            if col_offset + pan_window_width > global_width:
                patch_width = global_width - col_offset
            else:
                patch_width = pan_window_width

            if row_offset + pan_window_height > global_height:
                patch_height = global_height - row_offset
            else:
                patch_height = pan_window_height

            pan_window_config = {
                'col_off': col_offset,
                'row_off': row_offset,
                'width': patch_width,
                'height': patch_height
            }
            if include_partial_nodata:
                # Ignore patch that has only nodata. This mode is mainly used in inference phase.
                nodata_mask = get_nodata_mask(src_files, pan_window_config)
                if nodata_mask.sum() == nodata_mask.size:
                    continue
            else:
                # Ignore patch that has nodata. This mode is mainly used in training phase. 
                with rasterio.open(src_files[0]) as src:
                    nodata_mask = (src.read(1, window=Window(**pan_window_config))==0)
                if nodata_mask.sum() > 0:
                    continue

            window_configs.append(pan_window_config)

    return window_configs


def get_nodata_mask(src_files, pan_window_config):
    pan_file = src_files[0]
    ms_files = src_files[1:]
    ms_window_config = {
        'col_off': pan_window_config['col_off'] // 2,
        'row_off': pan_window_config['row_off'] // 2,
        'width': pan_window_config['width'] // 2,
        'height': pan_window_config['height'] // 2
    }

    with rasterio.open(pan_file) as src:
        nodata_mask = (src.read(1, window=Window(**pan_window_config)) == 0).astype(np.uint8)

    for ms_file in ms_files:
        with rasterio.open(ms_file) as src:
            ms_nodata_mask = (src.read(1, window=Window(**ms_window_config)) == 0).astype(np.uint8)
            upsampled_ms_nodata_mask = cv2.resize(
                ms_nodata_mask, 
                dsize=(nodata_mask.shape[1], nodata_mask.shape[0]), 
                interpolation=cv2.INTER_NEAREST
            )
            nodata_mask += upsampled_ms_nodata_mask

    nodata_mask = nodata_mask > 0

    return nodata_mask


def read_data(file_path, window_config):
    with rasterio.open(file_path) as src:
        arr = src.read(1, window=Window(**window_config))
        global_width = src.width
        global_height = src.height

    col_off = window_config['col_off']
    row_off = window_config['row_off']
    local_width = window_config['width']
    local_height = window_config['height']

    left_offset = 0
    top_offset = 0
    right_offset = 0
    bottom_offset = 0
    if col_off < 0:
        left_offset = abs(col_off)
    if row_off < 0:
        top_offset = abs(row_off)
    if col_off + local_width > global_width:
        right_offset = col_off + local_width - global_width
    if row_off + local_height > global_height:
        bottom_offset = row_off + local_height - global_height

    out = np.zeros((local_height, local_width))
    out[top_offset:local_height - bottom_offset, left_offset:local_width - right_offset] = arr
    out = out.astype(np.float32)

    return out

def crop_center(img, width, height):
    _, h, w = img.shape
    top = (h - height) // 2
    left = (w - width) // 2
    bottom = top + height
    right = left + width
    img = img[:, top:bottom, left:right]
    return img
