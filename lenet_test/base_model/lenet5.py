import torch
from torch import nn


class Lenet5(nn.Module):

    def __init__(self):
        super(Lenet5, self).__init__()

        # 卷积单元
        self.conv_unit = nn.Sequential(
            # x:[b, 3, 32, 32]
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        # 全连接单元
        self.fc_unit = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),  # 全连接之前有一个打平操作，所以输入为三者之积
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        # [b, 3, 32, 32] => [b, 16, 5, 5]
        batchsz = x.size(0)
        # batchsz = len(x)
        x = self.conv_unit(x)
        # [b, 16, 5, 5] => [b, 16*5*5]
        x = x.view(batchsz, -1)
        # [b, 16*5*5] => [b, 10]
        logits = self.fc_unit(x)
        return logits