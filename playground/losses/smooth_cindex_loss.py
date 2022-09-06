import torch.nn

import classic_algos.nn


class SmoothCindexLoss(classic_algos.nn.SmoothCindexLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, logits, labels, extras):
        return super().forward(logits, labels)
