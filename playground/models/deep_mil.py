import classic_algos.nn


class DeepMIL(classic_algos.nn.DeepMIL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, batch):
        return super().forward(batch["HISTO"][..., 3:], batch["HISTO_MASK"])[0]
