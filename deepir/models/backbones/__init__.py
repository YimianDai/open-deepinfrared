from .flex_resnet import FlexResNet
from .pvt_v2 import (FlexPVTv2, FlexPVTv2B0, FlexPVTv2B1, FlexPVTv2B2,
                     FlexPVTv2B2Li, FlexPVTv2B3, FlexPVTv2B4, FlexPVTv2B5)

__all__ = ['FlexResNet',
           'FlexPVTv2', 'FlexPVTv2B0', 'FlexPVTv2B1', 'FlexPVTv2B2',
           'FlexPVTv2B2Li', 'FlexPVTv2B3', 'FlexPVTv2B4', 'FlexPVTv2B5']