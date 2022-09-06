import torch.nn

import classic_algos.nn


class MSELoss(torch.nn.MSELoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, logits, labels):
        return super().forward(logits, labels)
