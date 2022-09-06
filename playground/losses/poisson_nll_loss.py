import torch.nn


class PoissonNLLLoss(torch.nn.PoissonNLLLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, logits, labels):
        return super().forward(logits, labels)
