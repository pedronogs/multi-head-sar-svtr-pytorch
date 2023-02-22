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
import torch


class CustomCTCLoss(nn.Module):

    def __init__(self, **kwargs):
        super(CustomCTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(blank=0, reduction='none')

    def forward(self, output, label, length):
        output = output.view((output.shape[1], output.shape[0], output.shape[2]))
        N, B, _ = output.shape
        preds_lengths = torch.tensor([N] * B, dtype=torch.long, device=torch.device('cpu'))
        labels = label.int()
        label_lengths = length.long()
        loss = self.loss_func(output, labels, preds_lengths, label_lengths)

        loss = loss.mean()

        return loss
