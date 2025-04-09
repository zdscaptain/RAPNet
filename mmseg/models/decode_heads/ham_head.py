# Copyright (c) OpenMMLab. All rights reserved.
# Originally from https://github.com/visual-attention-network/segnext
# Licensed under the Apache License, Version 2.0 (the "License")
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.device import get_device

from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead


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


@MODELS.register_module()
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
from .FreqFusion import FreqFusion
from .MADAttention import MFMSAttentionBlock
@MODELS.register_module()
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
                compress_ratio=8,
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
        self.MAD1 = MFMSAttentionBlock(in_channels=256, scale_branches=2, frequency_branches=16,
                                       frequency_selection='top', block_repetition=1)
        self.MAD2 = MFMSAttentionBlock(in_channels=160, scale_branches=2, frequency_branches=16,
                                       frequency_selection='top', block_repetition=1)
        self.MAD3 = MFMSAttentionBlock(in_channels=64, scale_branches=2, frequency_branches=16,
                                       frequency_selection='top', block_repetition=1)
        # self.MAD4 = MFMSAttentionBlock(in_channels=32, scale_branches=2, frequency_branches=16,
        #                                frequency_selection='top', block_repetition=1)
    def _forward_feature(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)
        # inputs = [resize(
        #     level,
        #     size=inputs[0].shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners
        # ) for level in inputs]

        # from low res to high res
        inputs = inputs[::-1]
        in_channels = self.in_channels[::-1]
        inputs[0] = self.MAD1(inputs[0])
        inputs[1] = self.MAD2(inputs[1])
        inputs[2] = self.MAD3(inputs[2])
        # inputs[3] = self.MAD4(inputs[3])
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
        x = self.squeeze(inputs)
        x = self.hamburger(x)
        output = self.align(x)

        # output = self.cls_seg(output)
        return output

    def forward(self, inputs):

        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output


