import torch.nn


class BCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, logits, labels, extras):
        return super().forward(logits, labels)
