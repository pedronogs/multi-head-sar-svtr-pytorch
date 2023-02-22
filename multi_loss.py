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

from ctc_loss import CustomCTCLoss
from sar_loss import CustomSARLoss


class MultiLoss(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.ctc_loss = CustomCTCLoss()
        self.sar_loss = CustomSARLoss()

    def forward(self, ctc_head_out, label_ctc, length, sar_head_out=None, label_sar=None):
        ctc_loss = self.ctc_loss(ctc_head_out, label_ctc, length)
        if self.training:
            return ctc_loss

        sar_loss = self.sar_loss(sar_head_out, label_sar, length)

        return ctc_loss, sar_loss
