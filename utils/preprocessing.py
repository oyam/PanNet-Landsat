import cv2
import numpy as np


def prepare_input(pan_arr, ms_arrs, use_original_scale=False):
    if use_original_scale:
        down_pan_shape = pan_arr.shape
        down_pan_arr = pan_arr
        down_ms_arrs = ms_arrs
    else:
        down_pan_shape = ms_arrs[0].shape
        down_ms_shape = tuple(np.array(down_pan_shape) // 2)
        down_pan_arr = _downsample(pan_arr, down_pan_shape)
        down_ms_arrs = [_downsample(arr, down_ms_shape) for arr in ms_arrs]

    # High pass filterd images
    hp_pan_arr = _high_pass_filter(down_pan_arr)
    hp_ms_arrs = \
        [_high_pass_filter(arr) for arr in down_ms_arrs]
    stacked_hp_up_ms = \
        np.stack([_upsample(arr, (down_pan_shape[1], down_pan_shape[0])) for arr in hp_ms_arrs])
    stacked_hp_pan_ms = np.vstack([hp_pan_arr[np.newaxis,:,:], stacked_hp_up_ms])

    # Spectra-mapping image
    stacked_up_ms = np.stack(
        [_upsample(arr, (down_pan_shape[1], down_pan_shape[0])) for arr in down_ms_arrs]
    )

    stacked_input = np.vstack((
        stacked_hp_pan_ms,
        stacked_up_ms
    ))

    return stacked_input


def _high_pass_filter(img, ksize=(5,5)):
    blur = cv2.blur(img, ksize)
    high_pass_filtered = img - blur
    return high_pass_filtered


def _downsample(img, dsize, ksize=(7,7), interpolation=cv2.INTER_AREA):
    blur = cv2.GaussianBlur(img, ksize, 0)
    downsampled = cv2.resize(blur, dsize, interpolation=interpolation)
    return downsampled


def _upsample(img, dsize, interpolation=cv2.INTER_CUBIC):
    upsampled = cv2.resize(img, dsize, interpolation=interpolation)
    return upsampled
