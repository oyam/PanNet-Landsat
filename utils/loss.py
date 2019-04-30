import chainer
import chainer.functions as F
from chainer import reporter


class MSELoss(chainer.Chain):

    def __init__(self, predictor):
        super(MSELoss, self).__init__()
        with self.init_scope():
            self.predictor = predictor

    def forward(self, x, t):
        """Computes the loss value for an image and label pair.
        Args:
            x (~chainer.Variable): A variable with a batch of images.
            t (~chainer.Variable): A variable with the ground truth.
        Returns:
            ~chainer.Variable: Loss value.
        """

        self.y = self.predictor(x)
        self.loss = F.mean_squared_error(self.y, t)
        mse_spectra_mapping_and_gt = F.mean_squared_error(x[:,4:,:,:], t)
        abs_error = F.mean_absolute_error(self.y, t)
        reporter.report({'loss': self.loss, 'mse_spectra_mapping_and_gt': mse_spectra_mapping_and_gt, 'absolute_error': abs_error}, self)
        return self.loss
