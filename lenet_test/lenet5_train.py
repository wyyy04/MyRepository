import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from lenet5 import Lenet5
from torch import nn, optim

def main():

    batchsz = 30
    # 读取数据，位置，是否为训练集，数据转化
    cifar_train = datasets.CIFAR10("cifar", True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()]), download=True)
    # 批次读取, 原数据，批次大小，是否乱序
    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)

    cifar_test = datasets.CIFAR10("cifar", False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)

    x, label = iter(cifar_train).next()  # 迭代器，一条一条的获取数据
    print("x, label", x.shape, label.shape)

    # device = torch.device("cuda")  # 设置加速器
    # model = Lenet5().to(device)  # 在载入模型

    model = Lenet5()  # 载入模型
    try:
        model.load_state_dict(torch.load('params.pkl'))
    except:
        print('')
    print(model)
    # criteon = nn.CrossEntropyLoss().to(device)  # 计算交叉熵
    criteon = nn.CrossEntropyLoss()  # 计算交叉熵
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # 设置优化器
    model.train()
    for epoch in range(1000):
        for batchid, (x, label) in enumerate(cifar_train):
            # x:[b, 3, 32, 32]
            # label:[b]
            # x, label = x.to(device), label.to(device)  # 转化到cuda上
            # logits:[b, 10]
            # label:[b]
            logits = model(x)  # 计算输出
            loss = criteon(logits, label)  # 计算损失
            # print(label.size(),logits.size())
            # backprob
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # loss更新梯度
            optimizer.step()  # 前向传播，更新权重

        print(epoch, loss.item())

        model.eval()
        with torch.no_grad():
            total_crrect = 0
            total_num = 0
            for x, label in cifar_test:
                # print((label))
                # x, label = x.to(device), label.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)  # 返回最大值所在位置
                total_crrect += torch.eq(pred, label).float().sum().item()  # 判断与label是否相等，返回正确个数
                total_num += x.size(0)
            acc = total_crrect / total_num  # 计算正确率
            print(epoch, acc)

        torch.save(model.state_dict(), 'params.pkl')
        model.load_state_dict(torch.load('params.pkl'))
        print()


if __name__ == '__main__':
    main()