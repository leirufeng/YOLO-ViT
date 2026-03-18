import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class MSCAM(nn.Module):
    def __init__(self, in_channels, out_channels, scales=[1, 2, 3], reduction_ratio=16):
        super(MSCAM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 1x1 卷积用于调整通道数，准备进行多尺度处理
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1x1 = nn.BatchNorm2d(out_channels)
        self.relu1x1 = nn.ReLU(inplace=True)

        # 多尺度卷积分支
        self.branches = nn.ModuleList()
        for scale in scales:
            # 对于每个尺度，使用不同的膨胀率（dilation）来模拟不同的感受野
            # 或者也可以使用不同大小的卷积核
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=scale, dilation=scale, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )

        # 融合多尺度特征的卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * len(scales), out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 通道注意力模块
        self.channel_attention = ChannelAttention(out_channels, reduction_ratio)

        # 最终的 1x1 卷积用于整合特征（如果需要进一步调整通道数）
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 1x1 卷积预处理
        x_prime = self.relu1x1(self.bn1x1(self.conv1x1(x)))

        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x_prime))

        # 拼接多尺度特征
        fused_features = torch.cat(branch_outputs, dim=1)

        # 融合卷积
        fused_features = self.fusion_conv(fused_features)

        # 应用通道注意力
        attention_map = self.channel_attention(fused_features)
        attended_features = fused_features * attention_map

        # 最终整合
        output = self.final_conv(attended_features)

        return output