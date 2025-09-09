# 怎么在gpu上训练？
# 找到->网络模型，数据（输入，标注），损失函数 .cuda()
import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import time
# from model import *
from torch.utils.data import DataLoader

# 准备数据集
train_data = torchvision.datasets.CIFAR10("dataset",train=True,transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10("dataset",train=False,transform=torchvision.transforms.ToTensor(),
                                          download=True)

# length长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集长度:{}".format(train_data_size))
print(f"测试数据集长度:{test_data_size}")

# 利用dataloader加载数据集
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)

# 创建网络模型
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10)
        )

    def forward(self, x):
        x = self.model(x)
        return x
tudui = Tudui()
if torch.cuda.is_available():
    tudui = tudui.cuda()

# 损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()
# 优化器
learning_rate = 1e-2# =0.01
optimizer = torch.optim.SGD(tudui.parameters(),lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 训练轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("logs_train")
start_time = time.time()

for i in range(epoch):
    print(f"---第 {i+1}--- 轮训练开始")

    # 训练步骤开始
    tudui.train()# 如果有特殊的层需要调用（看官网）
    for data in train_dataloader:
        imgs,targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(f"运行时间{end_time - start_time}")
            print(f"训练次数: {total_train_step} ，loss: {loss.item()}")
            writer.add_scalar("train_loss",loss.item(),total_train_step)


    # 测试步骤开始
    tudui.eval()
    total_test_loss = 0
    # 整体正确个数
    total_accuracy = 0
    with torch.no_grad():# 测试时不需要梯度
        for data in test_dataloader:
            imgs,targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = tudui(imgs)
            loss = loss_fn(outputs,targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print(f"整体测试集上的loss:{total_test_loss}")
    print(f"整体测试集上的正确率:{total_accuracy/test_data_size}")
    writer.add_scalar("test_loss",total_test_step,total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size,total_test_step)
    total_test_step += 1

    # torch.save(tudui,f"tudui_{i}.pth")
    torch.save(tudui.state_dict(),f"tudui_{i}.pth")
    print("模型已保存")

writer.close()