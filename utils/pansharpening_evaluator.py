import copy

from chainer import reporter
import chainer.training.extensions
from chainercv.utils import apply_to_iterator
import numpy as np


class PansharpeningEvaluator(chainer.training.extensions.Evaluator):

    trigger = 1, 'epoch'
    default_name = 'validation'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, iterator, target):
        super(PansharpeningEvaluator, self).__init__(
            iterator, target)

    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        in_values, out_values, rest_values = apply_to_iterator(
            target.predict, it)

        # delete unused iterators explicitly
        del in_values

        pred_imgs, = out_values
        gt_imgs, = rest_values

        mse_list = []
        for pred_img, gt_img in zip(pred_imgs, gt_imgs):
            diff = (pred_img - gt_img).ravel()
            mse = diff.dot(diff) / diff.size
            mse_list.append(mse)

        report = {
            'loss': np.mean(mse_list),
        }

        observation = {}
        with reporter.report_scope(observation):
            reporter.report(report, target)
        return observation
