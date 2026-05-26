from .srcnn import SRCNN
from .fsrcnn import FSRCNN
from .vdsr import VDSR
from .espcn import ESPCN
from .edsr import EDSR
from .rcan import RCAN
from .swinir import SwinIR

_REGISTRY = {
    'srcnn':  SRCNN,
    'fsrcnn': FSRCNN,
    'vdsr':   VDSR,
    'espcn':  ESPCN,
    'edsr':   EDSR,
    'rcan':   RCAN,
    'swinir': SwinIR,
}


def get_model(name: str, **kwargs):
    name = name.lower()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
