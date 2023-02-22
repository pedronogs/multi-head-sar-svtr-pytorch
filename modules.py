from torch import nn


class Im2Seq(nn.Layer):

    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == 1
        x = x.squeeze(axis=2)
        x = x.transpose([0, 2, 1])  # (NTC)(batch, width, channels)
        return x


class SequenceEncoder(nn.Layer):

    def __init__(self, in_channels, encoder_type, hidden_size=48, **kwargs):
        super(SequenceEncoder, self).__init__()
        self.encoder_reshape = Im2Seq(in_channels)
        self.out_channels = self.encoder_reshape.out_channels
        self.encoder_type = encoder_type

        self.encoder = EncoderWithSVTR(self.encoder_reshape.out_channels, **kwargs)

        self.out_channels = self.encoder.out_channels
        self.only_reshape = False

    def forward(self, x):
        x = self.encoder(x)
        x = self.encoder_reshape(x)
        return x


from ppocr.modeling.heads.rec_ctc_head import get_para_bias_attr
from paddleocr.modeling.backbones.rec_svtrnet import Block, ConvBNLayer, trunc_normal_, zeros_, ones_
import paddle

import math
from paddle import ParamAttr
from paddle.nn import functional as F


def get_para_bias_attr(l2_decay, k):
    regularizer = paddle.regularizer.L2Decay(l2_decay)
    stdv = 1.0 / math.sqrt(k * 1.0)
    initializer = nn.initializer.Uniform(-stdv, stdv)
    weight_attr = ParamAttr(regularizer=regularizer, initializer=initializer)
    bias_attr = ParamAttr(regularizer=regularizer, initializer=initializer)
    return [weight_attr, bias_attr]


class CTCHead(nn.Layer):

    def __init__(self, in_channels, out_channels, fc_decay=0.0004, mid_channels=None, return_feats=False, **kwargs):
        super(CTCHead, self).__init__()
        if mid_channels is None:
            weight_attr, bias_attr = get_para_bias_attr(l2_decay=fc_decay, k=in_channels)
            self.fc = nn.Linear(in_channels, out_channels, weight_attr=weight_attr, bias_attr=bias_attr)
        else:
            weight_attr1, bias_attr1 = get_para_bias_attr(l2_decay=fc_decay, k=in_channels)
            self.fc1 = nn.Linear(in_channels, mid_channels, weight_attr=weight_attr1, bias_attr=bias_attr1)

            weight_attr2, bias_attr2 = get_para_bias_attr(l2_decay=fc_decay, k=mid_channels)
            self.fc2 = nn.Linear(mid_channels, out_channels, weight_attr=weight_attr2, bias_attr=bias_attr2)
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.return_feats = return_feats

    def forward(self, x, targets=None):
        if self.mid_channels is None:
            predicts = self.fc(x)
        else:
            x = self.fc1(x)
            predicts = self.fc2(x)

        if self.return_feats:
            result = (x, predicts)
        else:
            result = predicts
        if not self.training:
            predicts = F.softmax(predicts, axis=2)
            result = predicts

        return result
