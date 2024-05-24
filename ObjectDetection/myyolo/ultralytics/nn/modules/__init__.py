# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f'{m._get_name()}.onnx'
    torch.onnx.export(m, x, f)
    os.system(f'onnxsim {f} {f} && open {f}')
    ```
"""

from .block import (C1, C2, C3, C3TR, DFL, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x, GhostBottleneck,
                    HGBlock, HGStem, Proto, RepC3,RepNCSPELAN4,
    ResNetLayer,
    Silence,   ADown,    CBLinear,    CBFuse,    SPPELAN,
)
from .conv import (CBAM, ChannelAttention, Concat, Conv, Conv2, ConvTranspose, DWConv, DWConvTranspose2d, Focus,
                   GhostConv, LightConv, RepConv, SpatialAttention, 
                   #SENetC3추가                
                   SENetC3)
from .head import OBB, Classify, Detect, Pose, RTDETRDecoder, Segment, WorldDetect
from .transformer import (AIFI, MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer, LayerNorm2d,
                          MLPBlock, MSDeformAttn, TransformerBlock, TransformerEncoderLayer, TransformerLayer,
                          #swin추가
                          MMlp, WindowAttention, SwinTransformerBlock ,PatchMerging, BasicLayer ,PatchEmbed, MyConcat6, MyConcat4, CST)

__all__ = ('Conv', 'Conv2', 'LightConv', 'RepConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus',
           'GhostConv', 'ChannelAttention', 'SpatialAttention', 'Concat', 'TransformerLayer',
           'TransformerBlock', 'MLPBlock', 'LayerNorm2d', 'DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3',
           'C2f', 'C3x', 'C3TR', 'C3Ghost', 'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'Detect',
           'Segment', 'Pose', 'Classify', 'TransformerEncoderLayer', 'RepC3', 'RTDETRDecoder', 'AIFI',
           'DeformableTransformerDecoder', 'DeformableTransformerDecoderLayer', 'MSDeformAttn', 'MLP',
           #추가모듈
           'MMlp', 'WindowAttention', 'SwinTransformerBlock' ,'PatchMerging', 'BasicLayer','PatchEmbed', 'CBAM','SENetC3', 'MyConcat6','MyConcat4', 'CST' ,"ResNetLayer",
    "OBB",
    "WorldDetect",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "RepNCSPELAN4",
    "ADown",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "Silence",
)