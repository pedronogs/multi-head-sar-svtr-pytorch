from torch import nn


class CustomSARLoss(nn.Module):

    def __init__(self, **kwargs):
        super(CustomSARLoss, self).__init__()
        ignore_index = kwargs.get('ignore_index', 92)  # 6626
        self.loss_func = nn.CrossEntropyLoss(reduction="mean", ignore_index=ignore_index)

    def forward(self, output, label, length):
        output = output[:, :-1, :]  # ignore last index of outputs to be in same seq_len with targets
        label = label.long()[:, 1:]  # ignore first index of target in loss calculation
        batch_size, num_steps, num_classes = output.shape[0], output.shape[1], output.shape[2]
        assert len(label.shape) == len(list(output.shape)) - 1, \
            "The target's shape and inputs's shape is [N, d] and [N, num_steps]"

        inputs = output.reshape([-1, num_classes])
        targets = label.reshape([-1])
        loss = self.loss_func(inputs, targets)
        return loss
