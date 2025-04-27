from .ce import ce_loss
from .triplet import triplet_loss
from .cw import cw_loss

loss_map = {
    "CE":      ce_loss,
    "Triplet": triplet_loss,
    "CW":      cw_loss,
}