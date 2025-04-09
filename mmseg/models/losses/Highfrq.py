# 新增对比损失类
import torch
import torch.nn as nn
import torch.nn.functional as F


class HighFreqContrastLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temp = temperature

    def forward(self, feat, target):
        # feat: 高频特征图 (B, C, H, W)
        # target: 二值掩码 (B, 1, H, W)
        pos_feat = feat * target  # 滑坡区域特征
        neg_feat = feat * (1 - target)  # 背景特征

        # 计算对比损失
        pos_mean = pos_feat.mean(dim=[2, 3])  # (B, C)
        neg_mean = neg_feat.mean(dim=[2, 3])
        sim = F.cosine_similarity(pos_mean, neg_mean, dim=1)  # (B,)
        loss = -torch.log(torch.exp(sim / self.temp)).mean()
        return loss

