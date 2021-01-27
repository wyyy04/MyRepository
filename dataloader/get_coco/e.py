# 导入相应的库
import torch
import time
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F

from dataloader import COCO_Dataloader
from torch.utils.data import DataLoader
import pandas as pd
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# 初始化设置
BATCH_SIZE = 8
num_epochs = 10

# 设置训练集
coco_dataloader = COCO_Dataloader()

cocom_train_data, cocom_test_data = coco_dataloader.get_dataset()
cocom_train = DataLoader(cocom_train_data, batch_size=BATCH_SIZE, shuffle=False)
cocom_test = DataLoader(cocom_test_data, batch_size=BATCH_SIZE, shuffle=False)
coco_test_labels = coco_dataloader.get_test_labels()

train_loader = cocom_train
test_loader = cocom_test

# 打印训练集测试集大小
train_data_size = len(cocom_train_data)
test_data_size = len(cocom_test_data)
print(train_data_size, test_data_size)



def evaluate(score,ground_truth):
    score = np.array((score))
    ground_truth = np.array(ground_truth)
    img_num,moti_num = score.shape
    score = pd.DataFrame(score)
    score = score.rank(method='first', axis=1)
    score = np.array(score)
    res = score[range(img_num), ground_truth]
    res = np.median(res)
    return res


# 构建Resnet模型
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=256):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])
def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])
def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])
def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])
def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


net = ResNet50().to('cuda:0')  # 设置为GPU训练

loss_function = nn.CrossEntropyLoss()  # 设置损失函数
optimizer = optim.Adam(net.parameters(), lr=0.0002)  # 设置优化器和学习率


# 定义训练函数
def train_and_valid(model, loss_function, optimizer, epochs=300):

    try:
        model.load_state_dict(torch.load('params.pkl'))
    except:
        print("No model exists.")
    res = np.zeros(test_data_size,256)
    mean_rank = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        model.eval()
        for j, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)

            outputs = model(inputs)
            res[j:j+BATCH_SIZE] = outputs.numpy()
        mean_rank = evaluate(res,coco_test_labels)
        print(mean_rank)

    return mean_rank


# 开始训练
res = train_and_valid(net, loss_function, optimizer, num_epochs)

