import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiScaleDilatedConv(nn.Module):
    def __init__(self, in_channels, dilations=[1, 2, 4]):
        super(MultiScaleDilatedConv, self).__init__()
        self.dilations = dilations
        self.num_scales = len(dilations)

        # 构造多尺度空洞卷积分支
        self.branches = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=d, padding=d, groups=in_channels, bias=False)
            for d in dilations
        ])

    def forward(self, x):
        # 对每个空洞卷积分支进行处理并返回多尺度特征
        out = sum(branch(x) for branch in self.branches) / self.num_scales
        return out


class FourierTransformAttention(nn.Module):
    def __init__(self, in_channels):
        super(FourierTransformAttention, self).__init__()
        self.in_channels = in_channels

        # 卷积来学习注意力权重
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        B, C, H, W = x.size()

        # 对输入进行傅里叶变换
        x_fft = torch.fft.fft2(x)
        x_fft = torch.fft.fftshift(x_fft)  # 频谱的零频成分移至中心

        # 获取高频信息
        high_freq = torch.abs(x_fft)  # 仅取幅度信息作为高频特征

        # 计算频域特征的注意力
        attn = torch.sigmoid(self.fc(high_freq))

        # 将注意力图与输入特征图相乘
        x = x * attn
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)


        self.fc = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        avg = self.avg_pool(x)
        max_ = self.max_pool(x)
        min_ = -self.max_pool(-x)

        # 将各个池化操作的结果拼接在一起，生成通道注意力图
        pooled = torch.cat([avg, max_, min_], dim=1)
        attn = torch.sigmoid(self.fc(pooled))
        return x * attn


class HFMSA(nn.Module):
    def __init__(self, in_channels, dilations=[1, 2, 4]):
        super(HFMSA, self).__init__()

        # 多尺度空洞卷积分支
        self.multi_scale_conv = MultiScaleDilatedConv(in_channels, dilations)

        # 频域增强
        self.fourier_attention = FourierTransformAttention(in_channels)

        # 通道注意力机制
        self.channel_attention = ChannelAttention(in_channels)

        # 最终的卷积融合
        self.fuse_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # 获取多尺度特征
        multi_scale_features = self.multi_scale_conv(x)

        # 进行频域增强
        enhanced_features = self.fourier_attention(multi_scale_features)

        # 应用通道注意力
        attention_features = self.channel_attention(enhanced_features)

        # 最终卷积融合
        out = self.fuse_conv(attention_features)

        # 残差连接
        return out + x


# ========== 测试代码 ==========

if __name__ == '__main__':
    input_tensor = torch.randn(1, 768, 8, 8)  # 示例输入

    # 创建HFMSA模块
    model = HFMSA(in_channels=768)

    # 获取输出
    output = model(input_tensor)

    print(f"Input size: {input_tensor.size()}")
    print(f"Output size: {output.size()}")
