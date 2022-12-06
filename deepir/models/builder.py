from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

MODELS = Registry('models', parent=MMCV_MODELS)

FUSES = MODELS
HEURISTICS = MODELS

def build_fuse(cfg):
    """Build attentional feature fusion methods."""
    return FUSES.build(cfg)

def build_heuristic(cfg):
    """Build heuristic methods."""
    return HEURISTICS.build(cfg)