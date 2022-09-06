import classic_algos.nn


class DeepMIL(classic_algos.nn.DeepMIL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, features):
        return super().forward(features["histo"][..., 3:], features["histo_mask"])[0]
