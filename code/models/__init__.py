from .srcnn import SRCNN
from .fsrcnn import FSRCNN
from .vdsr import VDSR
from .espcn import ESPCN
from .edsr import EDSR
from .rcan import RCAN
from .swinir import SwinIR
from .hat import HAT
from .drct import DRCT
from .ipt import IPT

_REGISTRY = {
    'srcnn':  SRCNN,
    'fsrcnn': FSRCNN,
    'vdsr':   VDSR,
    'espcn':  ESPCN,
    'edsr':   EDSR,
    'rcan':   RCAN,
    'swinir': SwinIR,
    'hat':    HAT,
    'drct':   DRCT,
    'ipt':    IPT,
}


# Models whose constructors do not accept a `scale` argument
# (they use pre-upsampled input; scale is handled in train.py/evaluate.py)
_NO_SCALE = {'srcnn', 'vdsr'}


def get_model(name: str, **kwargs):
    name = name.lower()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_REGISTRY)}")
    if name in _NO_SCALE:
        kwargs.pop('scale', None)
    return _REGISTRY[name](**kwargs)
