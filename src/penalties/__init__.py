from .kl import kl_pen
from .l2 import l2_pen
from .cosine import cos_pen

pen_map = {
    "KL":     kl_pen,
    "L2":     l2_pen,
    "Cosine": cos_pen,
}
