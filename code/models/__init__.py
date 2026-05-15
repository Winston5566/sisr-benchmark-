from .srcnn import SRCNN
from .fsrcnn import FSRCNN
from .vdsr import VDSR
from .espcn import ESPCN
from .edsr import EDSR

_REGISTRY = {
    'srcnn': SRCNN,
    'fsrcnn': FSRCNN,
    'vdsr': VDSR,
    'espcn': ESPCN,
    'edsr': EDSR,
}


def get_model(name: str, **kwargs):
    name = name.lower()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
