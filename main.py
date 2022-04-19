# _*_ coding:utf-8 _*_
# 系统提供的包
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np
# 自定义的包
from dataload import CovidCTDataset

# 按照通道标准化
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])

# 图像增广
train_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop((224), scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

val_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

batchsize = 32
total_epoch = 10

# 数据加载
trainset = CovidCTDataset(root_dir='images',
                          txt_COVID='data/COVID/trainCT_COVID.txt',
                          txt_NonCOVID='data/NonCOVID/trainCT_NonCOVID.txt',
                          transform=train_transformer)
valset = CovidCTDataset(root_dir='images',
                        txt_COVID='data/COVID/valCT_COVID.txt',
                        txt_NonCOVID='data/NonCOVID/valCT_NonCOVID.txt',
                        transform=val_transformer)
testset = CovidCTDataset(root_dir='images',
                         txt_COVID='data/COVID/testCT_COVID.txt',
                         txt_NonCOVID='data/NonCOVID/testCT_NonCOVID.txt',
                         transform=val_transformer)

print('训练集：', trainset.__len__())
print('验证集：', valset.__len__())
print('测试集：', testset.__len__())

# 构建DataLoader
from torch.utils.data import DataLoader
train_loader = DataLoader(trainset, batch_size=batchsize, drop_last=False, shuffle=True)
val_loader = DataLoader(valset, batch_size=batchsize, drop_last=False, shuffle=True)
test_loader = DataLoader(testset, batch_size=batchsize, drop_last=False, shuffle=True)

# 设置设备
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# 加载预训练模型
import torchxrayvision as xrv
model = xrv.models.DenseNet(num_classes=2, in_channels=3).to(device)
model_name = 'DenseNet_medical'
torch.cuda.empty_cache()

# 定义损失
criteria = nn.CrossEntropyLoss()
# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 使用设备训练
def train(optimizer, model, train_loader, criteria):
    model.train()
    train_loss = 0
    train_correct = 0

    for batch_index, batch_samples in enumerate(train_loader):
        # 将数据放到device
        data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
        # 前向传播
        output = model(data)
        # 计算损失
        loss = criteria(output, target.long())
        # 积累损失
        train_loss += loss
        # 清空上一轮梯度
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 参数更新
        optimizer.step()

        # 预测
        pred = output.argmax(dim=1, keepdim=True)
        # 累加预测与标签相等的次数
        train_correct += pred.eq(target.long().view_as(pred)).sum().item()

    # 显示训练结果
    print('训练集：平均损失:{:.4f}, 准确率：{}/{} ({:.0f}%)'.format(
        train_loss / len(train_loader.dataset),
        train_correct, len(train_loader.dataset),
        100 * train_correct / len(train_loader.dataset)))

    # 返回一次epoch的结果
    return train_loss / len(train_loader.dataset)


import torch.nn.functional as F
# 验证
def val(model, val_loader, criteria):
    # eval()
    model.eval()
    val_loss, correct = 0, 0

    # 不需要计算模型梯度
    with torch.no_grad():
        predlist, scorelist, targetlist = [], [], []
        # 预测
        for batch_index, batch_samples in enumerate(val_loader):
            data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
            # 前向传播
            output = model(data)
            val_loss += criteria(output, target.long())
            # 计算score
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.long().view_as(pred)).sum().item()

            # 放到CPU中
            predlist = np.append(predlist, pred.cpu().numpy())
            scorelist = np.append(scorelist, score.cpu().numpy()[:, 1])
            targetlist = np.append(targetlist, target.long().cpu().numpy())

    return targetlist, scorelist, predlist, val_loss / len(val_loader.dataset)


# 测试
def test(model, test_loader):
    # eval()
    model.eval()

    # 不需要计算模型梯度
    with torch.no_grad():
        predlist, scorelist, targetlist = [], [], []
        # 预测
        for batch_index, batch_samples in enumerate(test_loader):
            data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
            # 前向传播
            output = model(data)
            # 计算score
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)

            # 放到CPU中
            predlist = np.append(predlist, pred.cpu().numpy())
            scorelist = np.append(scorelist, score.cpu().numpy()[:, 1])
            targetlist = np.append(targetlist, target.long().cpu().numpy())

    return targetlist, scorelist, predlist


# 训练代码
for epoch in range(total_epoch):
    # 进行一次epoch
    train_loss = train(optimizer, model, train_loader, criteria)
    # 验证
    targetlist, scorelist, predlist, val_loss = val(model, val_loader, criteria)
    # 打印
    # print('Target:', targetlist)
    # # 输出label=1的概率
    # print('Score:', scorelist)
    # # 输出预测结果
    # print('Predict:', predlist)
    print(f'------ epoch: {epoch+1}/{total_epoch} finished ------')
    # 模型保存
    torch.save(model.state_dict(), 'covid_detection.pt')
