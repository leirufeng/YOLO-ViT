import torch
import torch.nn as nn


class FEM(nn.Module):
    def __init__(self, in_channels):
        super(FEM, self).__init__()
        # 1x1卷积，用于初步调整通道数
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 第一个分支：3x3标准卷积
        self.branch1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 第二个分支：包含空洞卷积的一系列卷积
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1,3), stride=1, padding=(0,1)),
            nn.Conv2d(in_channels, in_channels, kernel_size=(3,1), stride=1, padding=(1,0)),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=5, dilation=5),
        )
        # 第三个分支：与第二个分支结构类似，但卷积核顺序不同
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(3,1), stride=1, padding=(1,0)),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1,3), stride=1, padding=(0,1)),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=5, dilation=5),
        )
        self.cat = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, stride=1)

    def forward(self, x):
        # 1x1卷积调整通道数
        x_1x1 = self.conv1x1(x)
        # 各分支进行卷积操作
        branch1_out = self.branch1(x_1x1)
        branch2_out = self.branch2(x_1x1)
        branch3_out = self.branch3(x_1x1)
        # 特征图拼接
        cat_out = torch.cat([branch1_out, branch2_out, branch3_out], dim=1)
        # 与原始输入的1x1卷积结果相加
        out = self.cat(cat_out) + x_1x1
        return out


if __name__ == '__main__':
    x = torch.randn(4, 64, 128, 128).cuda()
    model = FEM(64).cuda()
    out = model(x)
    print(out.shape)
