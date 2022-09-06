# Classification losses
from .bce_with_logits_loss import BCEWithLogitsLoss
from .cross_entropy_loss import CrossEntropyLoss

# Regression losses
from .mse_loss import MSELoss
from .poisson_nll_loss import PoissonNLLLoss

# Survival losses
from .cox_loss import CoxLoss
from .smooth_cindex_loss import SmoothCindexLoss
