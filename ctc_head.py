# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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
import torch.nn.functional as F


class CTCHead(nn.Module):

    def __init__(self, in_channels, out_channels, fc_decay=0.0004, mid_channels=None, return_feats=False, **kwargs):
        super(CTCHead, self).__init__()
        if mid_channels is None:
            self.fc = nn.Linear(in_channels, out_channels, bias=True)
        else:
            self.fc1 = nn.Linear(in_channels, mid_channels, bias=True)
            self.fc2 = nn.Linear(mid_channels, out_channels, bias=True)

        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.return_feats = return_feats

    def forward(self, x):
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
            predicts = F.softmax(predicts, dim=2)
            result = predicts

        return result
