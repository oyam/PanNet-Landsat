import argparse
import os

import chainer
from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.training import extensions
import numpy as np

from pannet import PanNet
from utils import misc
from utils.landsat8_dataset import Landsat8Dataset
from utils.loss import MSELoss
from utils.pansharpening_evaluator import PansharpeningEvaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--alpha', type=float, default=0.001)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument('--weight_decay_rate', type=float, default=1e-4)
    parser.add_argument('--desable_amsgrad', dest='amsgrad', action='store_false')
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--out_model_dir', default='./out_model')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')
    parser.add_argument('--random_seed', default=1, type=int,
        help='Random seed for choosing cell ids and capture ids (default: 1)')
    args = parser.parse_args()

    os.makedirs(args.out_model_dir, exist_ok=True)

    # Download images
    os.makedirs(args.data_dir, exist_ok=True)
    baseurls = [
        'http://landsat-pds.s3.amazonaws.com/c1/L8/119/038/LC08_L1TP_119038_20171221_20171224_01_T1',
        'http://landsat-pds.s3.amazonaws.com/c1/L8/202/024/LC08_L1TP_202024_20181010_20181030_01_T1'
    ]
    for baseurl in baseurls:
        misc.download_landsat8_data(args.data_dir, baseurl)

    # Triggers
    log_trigger = (100, 'iteration')
    validation_trigger = (1, 'epoch')
    best_value_trigger = training.triggers.MaxValueTrigger(key='validation/main/loss', trigger=(1, 'epoch'))
    end_trigger = (args.epochs, 'epoch')

    # Dataset
    print('Preparing training data..')
    training_patch_configs, validation_patch_configs, _ =  \
        misc.get_patch_configs(
            args.data_dir, 
            training_data_rate=0.9,
            validation_data_rate=0.1,
            test_data_rate = 0.0,
            patch_width=64,
            patch_height=64,
            num_patches=18000,
            random_seed=args.random_seed
        )
    training_dataset = Landsat8Dataset(training_patch_configs)
    validation_dataset = Landsat8Dataset(validation_patch_configs)
    print('The number of training patches: ', len(training_dataset))
    print('The number of validation patches: ', len(validation_dataset))

    # Iterator
    train_iter = iterators.MultiprocessIterator(training_dataset, args.batch_size)
    val_iter = iterators.MultiprocessIterator(
        validation_dataset, 1, shuffle=False, repeat=False, dataset_timeout=None)

    # Model
    model = PanNet(out_channels=3)
    model = MSELoss(model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # Optimizer (AdamW with AMSGrad)
    optimizer = optimizers.Adam(alpha=args.alpha, beta1=args.beta1, beta2=args.beta2, eps=args.eps, eta=args.eta, weight_decay_rate=args.weight_decay_rate, amsgrad=args.amsgrad) 
    optimizer.setup(model)

    # Updater
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)

    # Trainer
    trainer = training.Trainer(updater, end_trigger, out=args.out_model_dir)

    trainer.extend(extensions.LogReport(trigger=log_trigger))
    trainer.extend(extensions.observe_lr(), trigger=log_trigger)
    trainer.extend(extensions.dump_graph('main/loss'))

    if args.plot and extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(
            ['main/loss', 'validation/main/loss'], x_key='iteration',
            file_name='loss.png'))

    trainer.extend(extensions.snapshot_object(
        model.predictor, filename='best_model'),
        trigger=best_value_trigger)
    trainer.extend(extensions.snapshot_object(
        model.predictor, filename='model_iteration-{.updater.iteration}'),
        trigger=end_trigger)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'elapsed_time', 'lr',
        'main/loss', 'main/mse_spectra_mapping_and_gt', 'main/absolute_error', 'validation/main/loss']),
        trigger=log_trigger)

    trainer.extend(
        PansharpeningEvaluator(val_iter, model.predictor),
        trigger=validation_trigger)

    print('Start training the model.')
    trainer.run()
    print('Finish')

if __name__ == '__main__':
    main()
