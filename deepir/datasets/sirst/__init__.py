from .cls2det import SIRSTCls2DetDataset
from .det2noco import SIRSTDet2NoCoDataset
from .noco import SIRSTNoCoDataset
from .seg import SIRSTSegDataset
from .seg2noco import SIRSTSeg2NoCoDataset

__all__ = ['SIRSTDet2NoCoDataset', 'SIRSTNoCoDataset', 'SIRSTSegDataset',
           'SIRSTSeg2NoCoDataset', 'SIRSTCls2DetDataset']