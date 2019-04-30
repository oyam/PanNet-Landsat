import chainer
import numpy as np

from . import misc
from . import preprocessing


class Landsat8Dataset(chainer.dataset.DatasetMixin):
    def __init__(self, patch_configs, use_original_scale=False):
        self.patch_configs = patch_configs
        self.use_original_scale = use_original_scale

    def __len__(self):
        return len(self.patch_configs)

    def get_example(self, i):
        if i >= len(self):
            raise IndexError('index is too large')

        patch_config = self.patch_configs.loc[i, :]
        src_files = patch_config['src_files']
        pan_window_config = patch_config['window_config']
        ms_window_config = {
            'col_off': pan_window_config['col_off'] // 2,
            'row_off': pan_window_config['row_off'] // 2,
            'width': pan_window_config['width'] // 2,
            'height': pan_window_config['height'] // 2
        }
        pan_arr = misc.read_data(src_files[0], pan_window_config)
        ms_arrs = [misc.read_data(f, ms_window_config) for f in src_files[1:]]

        stacked_input = preprocessing.prepare_input(pan_arr, ms_arrs, self.use_original_scale)
        ground_truth = np.stack(ms_arrs)

        return stacked_input, ground_truth
