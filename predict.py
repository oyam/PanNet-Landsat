import argparse
import os

import chainer

from pannet import PanNet
from utils import misc
from utils import pred_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--out_dir', default='./out')
    parser.add_argument('--pretrained_model', default='./out_model/best_model')
    parser.add_argument('--use_lower_scale', dest='use_original_scale', action='store_false')
    args = parser.parse_args()

    print('Use original scale: ', args.use_original_scale)

    print('Downloading data..')
    os.makedirs(args.data_dir, exist_ok=True)
    baseurls = [
        'http://landsat-pds.s3.amazonaws.com/c1/L8/119/038/LC08_L1TP_119038_20171221_20171224_01_T1',
        'http://landsat-pds.s3.amazonaws.com/c1/L8/202/024/LC08_L1TP_202024_20181010_20181030_01_T1'
    ]
    for baseurl in baseurls:
        misc.download_landsat8_data(args.data_dir, baseurl)

    print('Preparing patches..')
    _, _, patch_configs = misc.get_patch_configs(
        src_dir=args.data_dir, 
        training_data_rate=0.0, 
        validation_data_rate=0.0, 
        test_data_rate=1.0,
        patch_width=200, # Width of panchromatic will be 400 when you use original scale.
        patch_height=200, # Height of panchromatic will be 400 when you use original scale.
        num_patches=None, # Predict all patches in a file
        include_partial_nodata=True
    )
    print('The number of test patches: ', len(patch_configs))

    print('Loading model..')
    model = PanNet(out_channels=3)
    chainer.serializers.load_npz(args.pretrained_model, model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    print('Start prediction')
    pred_utils.predict_patches(
        model,
        patch_configs,
        pad=16,
        use_origial_scale=args.use_original_scale,
        dst_dir=args.out_dir, 
    )


if __name__ == '__main__':
    main()
