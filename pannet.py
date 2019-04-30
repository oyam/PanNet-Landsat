import chainer
import chainer.functions as F
import chainer.links as L
from chainercv.transforms import resize
import numpy as np


class PanNet(chainer.Chain):
    def __init__(self, out_channels=3):
        super(PanNet, self).__init__()
        with self.init_scope():
            self.first_conv = L.Convolution2D(None, 32, ksize=3, stride=1, pad=1)
            self.res = _ResBlock(32, ksize=3, stride=1, pad=1).repeat(4)
            self.last_conv = L.Convolution2D(32, out_channels, ksize=3, stride=1, pad=1)

    def forward(self, x):
        _, num_channels, _, _ = x.shape
        num_ms_channels = int((num_channels - 1) / 2)
        high_pass_pan_ms = x[:, :num_ms_channels+1, :, :]
        up_ms = x[:, num_ms_channels+1:, :, :]
        h = F.relu(self.first_conv(high_pass_pan_ms))
        h = self.res(h)
        h = self.last_conv(h) + up_ms
        return h

    def predict(self, imgs):
        outputs = []
        for img in imgs:
            C, H, W = img.shape
            with chainer.using_config('train', False), \
                    chainer.function.no_backprop_mode():
                x = chainer.Variable(self.xp.asarray(img[np.newaxis]))
                output = self.forward(x)[0].array
            output = chainer.backends.cuda.to_cpu(output)
            if output.shape != (C, H, W):
                dtype = output.dtype
                output = resize(output, (H, W)).astype(dtype)
            outputs.append(output)
        return outputs


class _ResBlock(chainer.Chain):
    def __init__(self, channels, ksize=3, stride=1, pad=1):
        super(_ResBlock, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(channels, channels, ksize=ksize, stride=stride, pad=pad)
            self.bn1 = L.BatchNormalization(channels)
            self.conv2 = L.Convolution2D(channels, channels, ksize=ksize, stride=stride, pad=pad)
            self.bn2 = L.BatchNormalization(channels)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(x)))
        h = h + x
        return h
