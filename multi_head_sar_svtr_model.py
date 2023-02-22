# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch import nn
from svtr_encoder import SVTREncoder
from util_layers import Im2Seq
from ctc_head import CTCHead
from sar_head import SARHead
from torchvision.models import regnet_y_800mf


class MultiHeadSARSVTRModel(nn.Module):

    def __init__(self, in_channels, n_characters, max_text_length=25):
        super().__init__()

        regnet = regnet_y_800mf(weights="RegNet_Y_800MF_Weights.IMAGENET1K_V1")
        self.backbone = nn.Sequential(regnet.stem, regnet.trunk_output)
        backbone_out_channels = 784

        self.sar_head = SARHead(
            in_channels=backbone_out_channels, out_channels=n_characters + 1, max_text_length=max_text_length
        )

        self.encoder_reshape = Im2Seq(backbone_out_channels)
        self.ctc_encoder = SVTREncoder(in_channels=backbone_out_channels)
        self.ctc_head = CTCHead(in_channels=self.ctc_encoder.out_channels, out_channels=n_characters)

    def forward(self, x, label=None):
        feats = self.backbone(x)
        feats = feats.view(feats.shape[0], feats.shape[1], 1, -1)

        ctc_encoder_out = self.ctc_encoder(feats)
        ctc_encoder_out = self.encoder_reshape(ctc_encoder_out)

        ctc_head_out = self.ctc_head(ctc_encoder_out)

        if not self.training:
            return ctc_head_out

        sar_head_out = self.sar_head(feats, label)

        return ctc_head_out, sar_head_out
