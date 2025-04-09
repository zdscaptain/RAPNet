# Copyright (c) OpenMMLab. All rights reserved.
# Originally from https://github.com/visual-attention-network/segnext
# Licensed under the Apache License, Version 2.0 (the "License")
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.device import get_device

from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead
from .module.FreqFusion import FreqFusion
from mmengine.visualization.visualizer import Visualizer
from .module.SAA import SAA,SCGA
from .module.MSA import MSAttention
from .module.FFCM import FFCM
from .module.Harr import HWD
from .module.SCAM import SCAM
from .module.TSCAM import TSCAM
from .module.CrossSparseAttention import CrossSparseAttention
# from .feedformer_head import FeedFormerHead_cat
import math
from timm.models.layers import  DropPath,trunc_normal_


class weight_cat(nn.Module):
    def __init__(self,dim=1,inchannel1=1,inchannel2=1):
        super(weight_cat,self).__init__()
        self.d = dim
        self.c1 = inchannel1
        self.c2 = inchannel2
        self.all_c = int(inchannel1+inchannel2)
        self.w = nn.Parameter(torch.ones(self.all_c, dtype=torch.float32), requires_grad=True)  # 可学习的权重参数，初始化为1
        self.b = nn.Parameter(torch.zeros(self.all_c, dtype=torch.float32), requires_grad=True)  # 可学习的偏置项
        self.epsilon = 0.0001  # 防止除零的小值

    def forward(self, x,y):
        N1, C1, H1, W1 = x.size()  # 获取第一个输入的维度
        N2, C2, H2, W2 = y.size()  # 获取第二个输入的维度

        w = self.w[:(C1 + C2)]  # 确保权重数量与输入通道数匹配
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 权重归一化
        x1 = (weight[:C1] * x.view(N1, H1, W1, C1) + self.b[:C1]).view(N1, C1, H1, W1)
        x2 = (weight[C1:] * y.view(N2, H2, W2, C2) + self.b[C1:]).view(N2, C2, H2, W2)


        x = torch.cat([x1,x2],self.d)
        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
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

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class CatKey(nn.Module):

    def __init__(self, pool_ratio, dim, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pool_ratio = pool_ratio
        self.pool1 = nn.AvgPool2d(self.pool_ratio[1], self.pool_ratio[1], ceil_mode=True)
        self.sr1 = nn.Conv2d(dim[1], dim[1], kernel_size=1, stride=1)
        self.pool2 = nn.AvgPool2d(self.pool_ratio[2], self.pool_ratio[2], ceil_mode=True)
        self.sr2 = nn.Conv2d(dim[2], dim[2], kernel_size=1, stride=1)
        self.pool3 = nn.AvgPool2d(self.pool_ratio[3], self.pool_ratio[3], ceil_mode=True)
        self.sr3 = nn.Conv2d(dim[3], dim[3], kernel_size=1, stride=1)
        # self.norm = nn.LayerNorm(dim2)
        # self.act = nn.GELU()

    def forward(self, x):
        return torch.cat([x[0], self.sr1(self.pool1(x[1])), self.sr2(self.pool2(x[2])), self.sr3(self.pool3(x[3]))],
                         dim=1)

class CatKey_harr(nn.Module):

    def __init__(self,  dim, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.harr1 = HWD(dim[1],dim[1])
        # self.pool1 = nn.AvgPool2d(self.pool_ratio[1], self.pool_ratio[1], ceil_mode=True)
        self.sr1 = nn.Conv2d(dim[1], dim[1], kernel_size=1, stride=1)
        self.harr2 = HWD(dim[2],dim[2])
        # self.pool2 = nn.AvgPool2d(self.pool_ratio[2], self.pool_ratio[2], ceil_mode=True)
        self.sr2 = nn.Conv2d(dim[2], dim[2], kernel_size=1, stride=1)
        # self.harr3 = HWD(dim[3],dim[3])
        # self.pool3 = nn.AvgPool2d(self.pool_ratio[3], self.pool_ratio[3], ceil_mode=True)
        # self.sr3 = nn.Conv2d(dim[3], dim[3], kernel_size=1, stride=1)
        # self.norm = nn.LayerNorm(dim2)
        # self.act = nn.GELU()

    def forward(self, x):
        x0,x1,x2 = x

        x1 = self.sr1(self.harr1(x1))
        x2 = self.harr2(x2)
        x2 = self.sr2(self.harr2(x2))
        # x3 = self.harr3(x3)
        # x3 = self.harr3(x3)
        # x3 = self.sr3(self.harr3(x3))
        x = torch.cat([x0,x1,x2],dim=1)


        return x
class CrossAttention(nn.Module):
    def __init__(self, dim1, dim2, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 pool_ratio=16):
        super().__init__()
        assert dim1 % num_heads == 0, f"dim {dim1} should be divided by num_heads {num_heads}."

        self.dim1 = dim1
        self.dim2 = dim2
        self.num_heads = num_heads
        head_dim = dim1 // num_heads
        self.pool_ratio = pool_ratio
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim1, dim1, bias=qkv_bias)
        self.kv = nn.Linear(dim2, dim1 * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim1, dim1)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.pool_ratio >= 0:
            self.pool = nn.AvgPool2d(self.pool_ratio, self.pool_ratio)
            self.sr = nn.Conv2d(dim2, dim2, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(dim2)
        self.act = nn.GELU()
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

    def forward(self, x, y, H2, W2):
        B1, N1, C1 = x.shape
        B2, N2, C2 = y.shape
        q = self.q(x).reshape(B1, N1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3)

        if self.pool_ratio >= 0:
            x_ = y.permute(0, 2, 1).reshape(B2, C2, H2, W2)
            x_ = self.sr(self.pool(x_)).reshape(B2, C2, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
        else:
            x_ = y

        kv = self.kv(x_).reshape(B1, -1, 2, self.num_heads, C1 // self.num_heads).permute(2, 0, 3, 1,
                                                                                          4)  # 여기에다가 rollout을 넣는다면?
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B1, N1, C1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):

    def __init__(self, dim1, dim2, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_ratio=16):
        super().__init__()
        self.norm1 = norm_layer(dim1)
        self.norm2 = norm_layer(dim2)
        self.norm3 = norm_layer(dim1)

        self.attn = CrossAttention(dim1=dim1, dim2=dim2, num_heads=num_heads, pool_ratio=pool_ratio)
        # self.csattn = CrossSparseAttention(in_channels1=dim1, in_channels2=dim2, num_heads=num_heads)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim1 * mlp_ratio)
        self.mlp = Mlp(in_features=dim1, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

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

    def forward(self, x, y, H2, W2, H1, W1):
        x = self.norm1(x)
        y = self.norm2(y)
        x = x + self.drop_path(self.attn(x, y, H2, W2)) #self.norm2(y)이 F1에 대한 값
        x = self.norm3(x)
        x = x + self.drop_path(self.mlp(x, H1, W1))

        # x = x + self.drop_path(self.attn(self.norm1(x), self.norm2(y), H2, W2)) #self.norm2(y)이 F1에 대한 값
        # x = x + self.drop_path(self.mlp(self.norm3(x), H1, W1))

        return x

# class CsBlock(nn.Module):
#
#     def __init__(self, dim1, dim2, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_ratio=16):
#         super().__init__()
#         self.norm1 = norm_layer(dim1)
#         self.norm2 = norm_layer(dim2)
#         self.norm3 = norm_layer(dim1)
#
#         self.attn = CrossAttention(dim1=dim1, dim2=dim2, num_heads=num_heads, pool_ratio=pool_ratio)
#         self.csattn = CrossSparseAttention(in_channels1=dim1, in_channels2=dim2, num_heads=num_heads)
#
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         mlp_hidden_dim = int(dim1 * mlp_ratio)
#         self.mlp = Mlp(in_features=dim1, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
#
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()
#
#     def forward(self, x, y, H2, W2, H1, W1):
#         x = self.norm1(x)
#         y = self.norm2(y)
#
#         x = x + self.drop_path(self.attn(x, y, H2, W2)) #self.norm2(y)이 F1에 대한 값
#         x = self.norm3(x)
#         x = x + self.drop_path(self.mlp(x, H1, W1))
#
#         # x = x + self.drop_path(self.attn(self.norm1(x), self.norm2(y), H2, W2)) #self.norm2(y)이 F1에 대한 값
#         # x = x + self.drop_path(self.mlp(self.norm3(x), H1, W1))
#
#         return x



class Matrix_Decomposition_2D_Base(nn.Module):
    """Base class of 2D Matrix Decomposition.

    Args:
        MD_S (int): The number of spatial coefficient in
            Matrix Decomposition, it may be used for calculation
            of the number of latent dimension D in Matrix
            Decomposition. Defaults: 1.
        MD_R (int): The number of latent dimension R in
            Matrix Decomposition. Defaults: 64.
        train_steps (int): The number of iteration steps in
            Multiplicative Update (MU) rule to solve Non-negative
            Matrix Factorization (NMF) in training. Defaults: 6.
        eval_steps (int): The number of iteration steps in
            Multiplicative Update (MU) rule to solve Non-negative
            Matrix Factorization (NMF) in evaluation. Defaults: 7.
        inv_t (int): Inverted multiple number to make coefficient
            smaller in softmax. Defaults: 100.
        rand_init (bool): Whether to initialize randomly.
            Defaults: True.
    """

    def __init__(self,
                 MD_S=1,
                 MD_R=64,
                 train_steps=6,
                 eval_steps=7,
                 inv_t=100,
                 rand_init=True):
        super().__init__()

        self.S = MD_S
        self.R = MD_R

        self.train_steps = train_steps
        self.eval_steps = eval_steps

        self.inv_t = inv_t

        self.rand_init = rand_init

    def _build_bases(self, B, S, D, R, device=None):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    def local_inference(self, x, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x, return_bases=False):
        """Forward Function."""
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B * S, D, N)
        D = C // self.S
        N = H * W
        x = x.view(B * self.S, D, N)
        if not self.rand_init and not hasattr(self, 'bases'):
            bases = self._build_bases(1, self.S, D, self.R, device=x.device)
            self.register_buffer('bases', bases)

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R, device=x.device)
        else:
            bases = self.bases.repeat(B, 1, 1)

        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        # (B * S, D, N) -> (B, C, H, W)
        x = x.view(B, C, H, W)

        return x


class NMF2D(Matrix_Decomposition_2D_Base):
    """Non-negative Matrix Factorization (NMF) module.

    It is inherited from ``Matrix_Decomposition_2D_Base`` module.
    """

    def __init__(self, args=dict()):
        super().__init__(**args)

        self.inv_t = 1

    def _build_bases(self, B, S, D, R, device=None):
        """Build bases in initialization."""
        if device is None:
            device = get_device()
        bases = torch.rand((B * S, D, R)).to(device)
        bases = F.normalize(bases, dim=1)

        return bases

    def local_step(self, x, bases, coef):
        """Local step in iteration to renew bases and coefficient."""
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = torch.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        """Compute coefficient."""
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # multiplication update
        coef = coef * numerator / (denominator + 1e-6)

        return coef


class Hamburger(nn.Module):
    """Hamburger Module. It consists of one slice of "ham" (matrix
    decomposition) and two slices of "bread" (linear transformation).

    Args:
        ham_channels (int): Input and output channels of feature.
        ham_kwargs (dict): Config of matrix decomposition module.
        norm_cfg (dict | None): Config of norm layers.
    """

    def __init__(self,
                 ham_channels=512,
                 ham_kwargs=dict(),
                 norm_cfg=None,
                 **kwargs):
        super().__init__()

        self.ham_in = ConvModule(
            ham_channels, ham_channels, 1, norm_cfg=None, act_cfg=None)

        self.ham = NMF2D(ham_kwargs)

        self.ham_out = ConvModule(
            ham_channels, ham_channels, 1, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x):
        enjoy = self.ham_in(x)
        enjoy = F.relu(enjoy, inplace=True)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        ham = F.relu(x + enjoy, inplace=True)

        return ham

# @MODELS.register_module()
class LightHamHead(BaseDecodeHead):
    """SegNeXt decode head.

    This decode head is the implementation of `SegNeXt: Rethinking
    Convolutional Attention Design for Semantic
    Segmentation <https://arxiv.org/abs/2209.08575>`_.
    Inspiration from https://github.com/visual-attention-network/segnext.

    Specifically, LightHamHead is inspired by HamNet from
    `Is Attention Better Than Matrix Decomposition?
    <https://arxiv.org/abs/2109.04553>`.

    Args:
        ham_channels (int): input channels for Hamburger.
            Defaults: 512.
        ham_kwargs (int): kwagrs for Ham. Defaults: dict().
    """

    def __init__(self, ham_channels=512, ham_kwargs=dict(), **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        self.ham_channels = ham_channels

        self.squeeze = ConvModule(
            sum(self.in_channels),
            self.ham_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.hamburger = Hamburger(ham_channels, ham_kwargs, **kwargs)

        self.align = ConvModule(
            self.ham_channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)


    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)
        inputs = [
            resize(
                level,
                size=inputs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners) for level in inputs
        ]

        inputs = torch.cat(inputs, dim=1)
        # apply a conv block to squeeze feature map
        x = self.squeeze(inputs)
        # apply hamburger module
        x = self.hamburger(x)

        # apply a conv block to align feature map
        output = self.align(x)
        output = self.cls_seg(output)
        return output
# @MODELS.register_module()
class FreqLightHamHead(BaseDecodeHead):


    def __init__(self, ham_channels=512, ham_kwargs=dict(), **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        self.ham_channels = ham_channels

        self.squeeze = ConvModule(
            sum(self.in_channels),
            self.ham_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        # self.squeeze = ConvModule(
        #     sum(self.in_channels * 2 ),
        #     self.ham_channels,
        #     1,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=self.norm_cfg,
        #     act_cfg=self.act_cfg)

        self.hamburger = Hamburger(ham_channels, ham_kwargs, **kwargs)

        self.align = ConvModule(
            self.ham_channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # self.saa0=FFCM(32)
        self.saa1=FFCM(self.in_channels[0])
        self.saa2=FFCM(self.in_channels[1])
        self.saa3=FFCM(self.in_channels[2])
        # self.crossatt = FeedFormerHead_cat(self.in_channels)
        # self.cat_k = CatKey(pool_ratio=[1,2,4,8],dim=[256,160,64,32])
        self.cat_k_harr = CatKey_harr(dim=[256,160,64])
        # # self.att4 = Block(dim1=self.in_channels[3],dim2=512,num_heads=8,mlp_ratio=4,drop_path=0.1,
        # #                   pool_ratio=8)
        self.att3 = Block(dim1=self.in_channels[2],dim2=sum(self.in_channels),num_heads=4,mlp_ratio=4,drop_path=0.1,
                          pool_ratio=4)
        self.att2 = Block(dim1=self.in_channels[1], dim2=sum(self.in_channels), num_heads=2, mlp_ratio=4, drop_path=0.1,
                          pool_ratio=2)
        self.att1 = Block(dim1=self.in_channels[0], dim2=sum(self.in_channels), num_heads=1, mlp_ratio=4, drop_path=0.1,
                          pool_ratio=1)
        self.linear_fuse = ConvModule(
            in_channels=sum(self.in_channels),
            out_channels=256,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )
        # self.csatt3 = CrossSparseAttention(dim1=self.in_channels[2],dim2=sum(self.in_channels),num_heads=8)
        # self.csatt2 = CrossSparseAttention(dim1=self.in_channels[1],dim2=sum(self.in_channels),num_heads=4)
        # self.csatt1 = CrossSparseAttention(dim1=self.in_channels[0],dim2=sum(self.in_channels),num_heads=2)
        self.scam1=SCAM(self.in_channels[0])
        self.scam2=SCAM(self.in_channels[1])
        self.scam3=SCAM(self.in_channels[2])
        self.scam4=SCAM(416)
        self.scam5=SCAM(416)

        self.ff1= FreqFusion(hr_channels=self.in_channels[1],lr_channels=self.in_channels[2])
        self.ff2= FreqFusion(hr_channels=self.in_channels[1],lr_channels=self.in_channels[2])
        self.ff3= FreqFusion(hr_channels=(self.in_channels[1]+self.in_channels[0]),lr_channels=(self.in_channels[2]+self.in_channels[1]))
        self.ff4= FreqFusion(hr_channels=(self.in_channels[0]),lr_channels=(self.in_channels[2]+self.in_channels[1]))

        self.wcat1=weight_cat(dim=1,inchannel1=self.in_channels[1],inchannel2=self.in_channels[2])
        self.wcat2=weight_cat(dim=1,inchannel1=self.in_channels[0],inchannel2=int(self.in_channels[2]+self.in_channels[1]))


    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)

        # 大残差连接
        # res = inputs

        # SCAM空间上下文
        inputs[0]=self.scam1(inputs[0])
        inputs[1]=self.scam2(inputs[1])
        inputs[2]=self.scam3(inputs[2])




        # freq fusion
        _,feature1,feature2 = self.ff1(inputs[1],inputs[2])
        feature3 = self.wcat1(feature1,feature2)
        feature3 = self.scam4(feature3)
        _,f4,f5 = self.ff4(inputs[0],feature3)
        x = self.wcat2(f4,f5)
        # feature3 = self.scam4(feature3)
        # _,feature4,feature5 = self.ff2(inputs[1],inputs[2])
        # feature6 = torch.cat([feature5,feature4],dim=1)
        # feature6 = self.scam5(feature6)
        # _,x1,x2 = self.ff3(feature3,feature6)
        # x = torch.cat([x1,x2],dim=1)



        # FFCM空间特征频率域提取
        # inputs[0] = self.saa0(inputs[0])+inputs[0]

        # c1,c2,c3 = inputs
        # _c3 = self.csatt3(c3,c_k)
        # _c2 = self.csatt2(c2,c_k)
        # _c1 = self.csatt1(c1,c_k)

        # c1 = inputs[0].flatten(2).transpose(1, 2)
        # c2 = inputs[1].flatten(2).transpose(1, 2)
        # c3 = inputs[2].flatten(2).transpose(1, 2)
        # # # c4 = inputs[3].flatten(2).transpose(1, 2)  # shape: [batch, h1*w1, patches]
        # c_key = c_k.flatten(2).transpose(1, 2)  # shape: [batch, h1*w1, patches]
        #
        # # _c4 = self.att4(c4, c_key, h4, w4, h4, w4)
        # # _c4 += c4
        # # _c4 = _c4.permute(0, 2, 1).reshape(n, -1, h4, w4)
        # # _c4 = resize(_c4, size=(h1, w1), mode='bilinear', align_corners=False)
        #
        # _c3 = self.att3(c3, c_key, h3, w3, h3, w3)
        # # # _c3 += c3
        # _c3 = _c3.permute(0, 2, 1).reshape(n, -1, h3, w3)
        # # _c3 = resize(_c3, size=(h1, w1), mode='bilinear', align_corners=False)
        # #
        # _c2 = self.att2(c2, c_key, h3, w3, h2, w2)
        # # # _c2 += c2
        # _c2 = _c2.permute(0, 2, 1).reshape(n, -1, h2, w2)
        # # _c2 = resize(_c2, size=(h1, w1), mode='bilinear', align_corners=False)
        # #
        # _c1 = self.att1(c1, c_key, h3, w3, h1, w1)
        # _c1 = c1.permute(0, 2, 1).reshape(n, -1, h1, w1)
        # # _c1 = resize(_c1, size=(h1, w1), mode='bilinear', align_corners=False)
        #
        # # x = self.linear_fuse(self.cat_k_harr([_c3, _c2, _c1]))
        # x = self.cat_k_harr([_c3,_c2,_c1])
        # x = self.linear_fuse(x)


        # inputs = [
        #     resize(
        #         level,
        #         size=inputs[0].shape[2:],
        #         mode='bilinear',
        #         align_corners=self.align_corners) for level in inputs
        # ]

        # # 大残差连接
        # res = [
        #     resize(
        #         level,
        #         size=res[0].shape[2:],
        #         mode='bilinear',
        #         align_corners=self.align_corners) for level in res
        # ]
        # res = torch.cat(res,dim=1)
        # x = torch.cat([x,res],dim=1)

        #
        # inputs = torch.cat(inputs, dim=1)
        # x = self.squeeze(inputs)
        # apply a conv block to squeeze feature map
        x = self.squeeze(x)
        # apply hamburger module
        x = self.hamburger(x)

        # apply a conv block to align feature map
        output = self.align(x)
        output = self.cls_seg(output)
        return output


@MODELS.register_module()
class FreqLightHamHead1(BaseDecodeHead):

    def __init__(self, ham_channels=512, ham_kwargs=dict(), **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        self.ham_channels = ham_channels

        self.squeeze = ConvModule(
            sum(self.in_channels),
            self.ham_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        # self.squeeze = ConvModule(
        #     sum(self.in_channels * 2 ),
        #     self.ham_channels,
        #     1,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=self.norm_cfg,
        #     act_cfg=self.act_cfg)

        self.hamburger = Hamburger(ham_channels, ham_kwargs, **kwargs)

        self.align = ConvModule(
            self.ham_channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.scam1 = TSCAM(self.in_channels[0])
        self.scam2 = TSCAM(self.in_channels[1])
        self.scam3 = TSCAM(self.in_channels[2])


        self.ff1 = FreqFusion(hr_channels=self.in_channels[1], lr_channels=self.in_channels[2])
        self.ff2 = FreqFusion(hr_channels=self.in_channels[0], lr_channels=self.in_channels[1])
        self.ff3 = FreqFusion(hr_channels=self.in_channels[0], lr_channels=self.in_channels[2])

    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)

        # 大残差连接
        # res = inputs

        # SCAM空间上下文
        inputs[0] = self.scam1(inputs[0])
        inputs[1] = self.scam2(inputs[1])
        inputs[2] = self.scam3(inputs[2])

        # freq fusion
        _, feature1, feature2 = self.ff1(inputs[1], inputs[2])
        _, feature3, feature4 = self.ff2(inputs[0], feature1)
        _, feature6, feature7 = self.ff3(inputs[0], feature2)
        feature8 = feature3+feature6+inputs[0]
        x = torch.cat([feature4,feature7,feature8],dim=1)




        # inputs = [
        #     resize(
        #         level,
        #         size=inputs[0].shape[2:],
        #         mode='bilinear',
        #         align_corners=self.align_corners) for level in inputs
        # ]


        x = self.squeeze(x)
        # apply hamburger module
        x = self.hamburger(x)

        # apply a conv block to align feature map
        output = self.align(x)
        output = self.cls_seg(output)
        return output



# @MODELS.register_module()
class LightHamHeadFreqAware(LightHamHead):
    """Is Attention Better Than Matrix Decomposition?
    This head is the implementation of `HamNet
    <https://arxiv.org/abs/2109.04553>`_.
    Args:
        ham_channels (int): input channels for Hamburger.
        ham_kwargs (int): kwagrs for Ham.

    TODO:
        Add other MD models (Ham).
    """

    def __init__(self,
                use_high_pass=True,
                use_low_pass=True,
                compress_ratio=4,
                semi_conv=True,
                low2high_residual=False,
                high2low_residual=False,
                lowpass_kernel=5,
                highpass_kernel=3,
                hamming_window=False,
                feature_resample=True,
                feature_resample_group=4,
                comp_feat_upsample=True,
                use_checkpoint=False,
                feature_resample_norm=True,
                **kwargs):
        super().__init__(**kwargs)
        self.freqfusions = nn.ModuleList()
        in_channels = kwargs.get('in_channels', [])
        self.feature_resample = feature_resample
        self.feature_resample_group = feature_resample_group
        self.use_checkpoint = use_checkpoint
        # from lr to hr
        in_channels = in_channels[::-1]
        pre_c = in_channels[0]
        for c in in_channels[1:]:
            freqfusion = FreqFusion(
                hr_channels=c, lr_channels=pre_c, scale_factor=1, lowpass_kernel=lowpass_kernel, highpass_kernel=highpass_kernel, up_group=1,
                upsample_mode='nearest', align_corners=False,
                feature_resample=feature_resample, feature_resample_group=feature_resample_group,
                comp_feat_upsample=comp_feat_upsample,
                hr_residual=True,
                hamming_window=hamming_window,
                compressed_channels= (pre_c + c) // compress_ratio,
                use_high_pass=use_high_pass, use_low_pass=use_low_pass, semi_conv=semi_conv,
                feature_resample_norm=feature_resample_norm,
                )
            self.freqfusions.append(freqfusion)
            pre_c += c

        # from lr to hr
        assert not (low2high_residual and high2low_residual)
        self.low2high_residual = low2high_residual
        self.high2low_residual = high2low_residual
        if low2high_residual:
            self.low2high_convs = nn.ModuleList()
            pre_c = in_channels[0]
            for c in in_channels[1:]:
                self.low2high_convs.append(nn.Conv2d(pre_c, c, 1))
                pre_c = c
        elif high2low_residual:
            self.high2low_convs = nn.ModuleList()
            pre_c = in_channels[0]
            for c in in_channels[1:]:
                self.high2low_convs.append(nn.Conv2d(c, pre_c, 1))
                pre_c += c
        # self.save = feature_vis()
        self.scam1 = SCAM(self.in_channels[0])
        self.scam2 = SCAM(self.in_channels[1])
        self.scam3 = SCAM(self.in_channels[2])

    def _forward_feature(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)

        # 空间上下文SCAM
        # inputs[0] = self.scam1(inputs[0])
        # inputs[1] = self.scam2(inputs[1])
        # inputs[2] = self.scam3(inputs[2])

        # inputs = [resize(
        #     level,
        #     size=inputs[0].shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners
        # ) for level in inputs]

        # from low res to high res
        inputs = inputs[::-1]
        in_channels = self.in_channels[::-1]
        lowres_feat = inputs[0]
        if self.low2high_residual:
            for pre_c, hires_feat, freqfusion, low2high_conv in zip(in_channels[:-1], inputs[1:], self.freqfusions, self.low2high_convs):
                _, hires_feat, lowres_feat = freqfusion(hr_feat=hires_feat, lr_feat=lowres_feat, use_checkpoint=self.use_checkpoint)
                lowres_feat = torch.cat([hires_feat + low2high_conv(lowres_feat[:, :pre_c]), lowres_feat], dim=1)
            pass
        else:
            for idx, (hires_feat, freqfusion) in enumerate(zip(inputs[1:], self.freqfusions)):
                _, hires_feat, lowres_feat = freqfusion(hr_feat=hires_feat, lr_feat=lowres_feat, use_checkpoint=self.use_checkpoint)
                if self.feature_resample:
                    b, _, h, w = hires_feat.shape
                    lowres_feat = torch.cat([hires_feat.reshape(b * self.feature_resample_group, -1, h, w),
                                             lowres_feat.reshape(b * self.feature_resample_group, -1, h, w)], dim=1).reshape(b, -1, h, w)
                else:
                    lowres_feat = torch.cat([hires_feat, lowres_feat], dim=1)

        # inputs = torch.cat(inputs, dim=1)
        inputs = lowres_feat
        # featmap = inputs
        # B, C, H, W = featmap.shape
        # featmap_3d = featmap.view(B * C, H, W)  # 将 batch 和通道维度合并
        # featmap_3d 的形状为 (B*C, H, W)
        # vis = Visualizer()
        # featshow=vis.draw_featmap(featmap_3d)
        # cv2.imshow('Feature Map', featshow)
        # cv2.waitKey(0)  # 按任意键退出
        # cv2.destroyAllWindows()

        x = self.squeeze(inputs)
        x = self.hamburger(x)
        output = self.align(x)

        # output = self.cls_seg(output)
        return output

    def forward(self, inputs):
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)


        return output