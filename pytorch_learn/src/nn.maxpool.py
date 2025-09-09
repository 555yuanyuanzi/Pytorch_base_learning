# 最大池化层：降低特征的数据量
import torch
import torchvision
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("dataset",train=False, download=False,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1 ,2 ,1 ,0, 0],
#                       [5, 2 ,3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype=torch.float32)
# input = torch.reshape(input,(-1, 1, 5, 5))
# print(input.shape)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)


    def forward(self, input):
        output = self.maxpool1(input)
        return  output

tudui = Tudui()

writer= SummaryWriter("logs_maxpool")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input",imgs,step)
    output = tudui(imgs)# 池化不改变channel数
    writer.add_images("output", output, step)

    step = step + 1

writer.close()