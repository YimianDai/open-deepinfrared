import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from mmdet.models.builder import BACKBONES
from mmdet.utils import get_root_logger
from mmcv.runner import load_checkpoint
import math

from mmseg.models.builder import BACKBONES as SEGBACKBONES
from mmdet.models.builder import BACKBONES as DETBACKBONES

from deepir.models.utils import FlexBlock


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self,
                 patch_size=7,
                 stride=4,
                 in_channels=3,
                 embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        assert max(patch_size) > stride, "Set larger patch_size than stride"
        self.proj = nn.Conv2d(in_channels,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


@DETBACKBONES.register_module()
@SEGBACKBONES.register_module()
class FlexPVTv2(nn.Module):
    """Flexible Pyramid Vision Transformer v2 backbone for DeepInfrared.
        Default: PVTv2B2

    Args:
        depths (Sequence[int]): layers for stages of PVTv2.
            Default: [3, 3, 6, 3]
        embed_dims (Sequence[int]): Output channel numbers of each stage.
            Default: [64, 128, 320, 512].
        strides (Sequence[int]): Strides of the first block of each stage.
            Default: [4, 2, 2, 2].
        num_heads (Sequence[int]): Head number of the Efficient Self-Attention
            in each stage. Default: [1, 2, 5, 8].
        mlp_ratios (Sequence[int]): Expansion ratio of the feed-forward layer
            in each stage. Default: [8, 8, 4, 4].
        sr_ratios (Sequence[int]): Reduction ratio of the SRA in each stage.
            Default: [8, 4, 2, 1].
        linear (bool): Whether to use Linear SRA. Default: False.
        pooling_sizes: Adaptive average pooling size of the Linear SRA in each
            stage. Default: [7, 7, 7, 7].
        in_channels (int): Number of input image channels. Default: 3.
        patch_sizes (Sequence[int]): patch sizes for each stage, similar to
            the kernel size of convolutional layers in ResNet.
        TODO:
        qkv_bias
        qk_scale
        drop_rate
        attn_drop_rate
        drop_path_rate
        norm_layer
        pretrained
    """
    def __init__(self,
                 depths=[3, 3, 6, 3], # L_i
                 embed_dims=[64, 128, 320, 512], # C_i
                 strides=(4, 2, 2, 2), # S_i
                 num_heads=[1, 2, 5, 8], # N_i
                 mlp_ratios=[8, 8, 4, 4], # E_i
                 sr_ratios=[8, 4, 2, 1], # R_i
                 linear=False,
                 pooling_sizes=[7, 7, 7, 7], # P_i
                 in_channels=3,
                 patch_sizes=[7, 3, 3, 3],
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 pretrained=None,
                 **kwargs):
        super(FlexPVTv2, self).__init__(**kwargs)
        self.depths = depths
        assert len(depths) == len(strides) == len(patch_sizes) == len(embed_dims)
        self.num_stages = len(depths)
        self.linear = linear
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(self.num_stages):
            inplanes = in_channels if i == 0 else embed_dims[i-1]
            patch_embed = OverlapPatchEmbed(patch_size=patch_sizes[i],
                                            stride=strides[i],
                                            in_channels=inplanes,
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([
                FlexBlock(dim=embed_dims[i],
                          num_heads=num_heads[i],
                          mlp_ratio=mlp_ratios[i],
                          qkv_bias=qkv_bias,
                          qk_scale=qk_scale,
                          drop=drop_rate,
                          attn_drop=attn_drop_rate,
                          drop_path=dpr[cur + j],
                          norm_layer=norm_layer,
                          sr_ratio=sr_ratios[i],
                          linear=linear,
                          pooling_size=pooling_sizes[i])
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        self.apply(self._init_weights)
        self.init_weights(pretrained)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4',
                'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs


    def forward(self, x):
        x = self.forward_features(x)

        return x


@DETBACKBONES.register_module()
@SEGBACKBONES.register_module()
class FlexPVTv2B0(FlexPVTv2):
    """ fix the following settings:
        ```
        depths=[2, 2, 2, 2],
        linear=False
        ```
    """
    def __init__(self,
                 strides=(4, 2, 2, 2), # S_i
                 embed_dims=[32, 64, 160, 256], # C_i
                 num_heads=[1, 2, 5, 8], # N_i
                 mlp_ratios=[8, 8, 4, 4], # E_i
                 sr_ratios=[8, 4, 2, 1], # R_i
                 in_channels=3,
                 patch_sizes=[7, 3, 3, 3],
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 pretrained=None,
                 **kwargs):
        super(FlexPVTv2B0, self).__init__(
            depths=[2, 2, 2, 2], # L_i
            embed_dims=embed_dims, # C_i
            strides=strides,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            sr_ratios=sr_ratios,
            linear=False,
            in_channels=in_channels,
            patch_sizes=patch_sizes,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            pretrained=pretrained)


@DETBACKBONES.register_module()
@SEGBACKBONES.register_module()
class FlexPVTv2B1(FlexPVTv2):
    """ fix the following settings:
        ```
        depths=[2, 2, 2, 2],
        linear=False
        ```
    """
    def __init__(self,
                 strides=(4, 2, 2, 2), # S_i
                 embed_dims=[64, 128, 320, 512], # C_i
                 num_heads=[1, 2, 5, 8], # N_i
                 mlp_ratios=[8, 8, 4, 4], # E_i
                 sr_ratios=[8, 4, 2, 1], # R_i
                 in_channels=3,
                 patch_sizes=[7, 3, 3, 3],
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 pretrained=None,
                 **kwargs):
        super(FlexPVTv2B1, self).__init__(
            depths=[2, 2, 2, 2], # L_i
            embed_dims=embed_dims,
            strides=strides,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            sr_ratios=sr_ratios,
            linear=False,
            in_channels=in_channels,
            patch_sizes=patch_sizes,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            pretrained=pretrained)


@DETBACKBONES.register_module()
@SEGBACKBONES.register_module()
class FlexPVTv2B2(FlexPVTv2):
    """ fix the following settings:
        ```
        depths=[3, 3, 6, 3],
        linear=False
        ```
    """
    def __init__(self,
                 strides=(4, 2, 2, 2), # S_i
                 embed_dims=[64, 128, 320, 512], # C_i
                 num_heads=[1, 2, 5, 8], # N_i
                 mlp_ratios=[8, 8, 4, 4], # E_i
                 sr_ratios=[8, 4, 2, 1], # R_i
                 in_channels=3,
                 patch_sizes=[7, 3, 3, 3],
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 pretrained=None,
                 **kwargs):
        super(FlexPVTv2B2, self).__init__(
            depths=[3, 3, 6, 3], # L_i
            embed_dims=embed_dims, # C_i
            strides=strides,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            sr_ratios=sr_ratios,
            linear=False,
            in_channels=in_channels,
            patch_sizes=patch_sizes,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            pretrained=pretrained)


@DETBACKBONES.register_module()
@SEGBACKBONES.register_module()
class FlexPVTv2B2Li(FlexPVTv2):
    """ fix the following settings:
        ```
        depths=[3, 3, 6, 3],
        linear=True
        ```
    """
    def __init__(self,
                 strides=(4, 2, 2, 2), # S_i
                 embed_dims=[64, 128, 320, 512], # C_i
                 num_heads=[1, 2, 5, 8], # N_i
                 mlp_ratios=[8, 8, 4, 4], # E_i
                 sr_ratios=[8, 4, 2, 1], # R_i
                 pooling_sizes=[7, 7, 7, 7], # P_i
                 in_channels=3,
                 patch_sizes=[7, 3, 3, 3],
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 pretrained=None,
                 **kwargs):
        super(FlexPVTv2B2Li, self).__init__(
            depths=[3, 3, 6, 3], # L_i
            embed_dims=embed_dims, # C_i
            strides=strides,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            sr_ratios=sr_ratios,
            pooling_sizes=pooling_sizes,
            linear=True,
            in_channels=in_channels,
            patch_sizes=patch_sizes,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            pretrained=pretrained)


@DETBACKBONES.register_module()
@SEGBACKBONES.register_module()
class FlexPVTv2B3(FlexPVTv2):
    """ fix the following settings:
        ```
        depths=[3, 3, 18, 3],
        linear=False
        ```
    """
    def __init__(self,
                 strides=(4, 2, 2, 2), # S_i
                 embed_dims=[64, 128, 320, 512], # C_i
                 num_heads=[1, 2, 5, 8], # N_i
                 mlp_ratios=[8, 8, 4, 4], # E_i
                 sr_ratios=[8, 4, 2, 1], # R_i
                 in_channels=3,
                 patch_sizes=[7, 3, 3, 3],
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 pretrained=None,
                 **kwargs):
        super(FlexPVTv2B3, self).__init__(
            depths=[3, 3, 18, 3], # L_i
            embed_dims=embed_dims, # C_i
            strides=strides,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            sr_ratios=sr_ratios,
            linear=False,
            in_channels=in_channels,
            patch_sizes=patch_sizes,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            pretrained=pretrained)


@DETBACKBONES.register_module()
@SEGBACKBONES.register_module()
class FlexPVTv2B4(FlexPVTv2):
    """ fix the following settings:
        ```
        depths=[3, 8, 27, 3],
        linear=False
        ```
    """
    def __init__(self,
                 strides=(4, 2, 2, 2), # S_i
                 embed_dims=[64, 128, 320, 512], # C_i
                 num_heads=[1, 2, 5, 8], # N_i
                 mlp_ratios=[8, 8, 4, 4], # E_i
                 sr_ratios=[8, 4, 2, 1], # R_i
                 in_channels=3,
                 patch_sizes=[7, 3, 3, 3],
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 pretrained=None,
                 **kwargs):
        super(FlexPVTv2B4, self).__init__(
            depths=[3, 8, 27, 3], # L_i
            embed_dims=embed_dims, # C_i
            strides=strides,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            sr_ratios=sr_ratios,
            linear=False,
            in_channels=in_channels,
            patch_sizes=patch_sizes,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            pretrained=pretrained)


@DETBACKBONES.register_module()
@SEGBACKBONES.register_module()
class FlexPVTv2B5(FlexPVTv2):
    """ fix the following settings:
        ```
        depths=[3, 6, 40, 3],
        linear=False
        ```
    """
    def __init__(self,
                 strides=(4, 2, 2, 2), # S_i
                 embed_dims=[64, 128, 320, 512], # C_i
                 num_heads=[1, 2, 5, 8], # N_i
                 mlp_ratios=[4, 4, 4, 4], # E_i
                 sr_ratios=[8, 4, 2, 1], # R_i
                 in_channels=3,
                 patch_sizes=[7, 3, 3, 3],
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 pretrained=None,
                 **kwargs):
        super(FlexPVTv2B5, self).__init__(
            depths=[3, 6, 40, 3], # L_i
            embed_dims=embed_dims, # C_i
            strides=strides,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            sr_ratios=sr_ratios,
            linear=False,
            in_channels=in_channels,
            patch_sizes=patch_sizes,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            pretrained=pretrained)