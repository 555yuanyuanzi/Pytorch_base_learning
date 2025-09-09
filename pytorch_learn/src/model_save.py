import torch
import torchvision

import torch
import torchvision
from torch import nn

# 加载模型并获取预训练权重
model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
# 保存方式1 模型结构+参数
torch.save(model, "vgg16_method1.pth")# PYTORCH网络模型保存格式

# 保存方式2 模型参数（官方推荐）
torch.save(model.state_dict(),"vgg16_method2.pth")

# 陷阱
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(3,64,3)

    def forward(self, x):
        x = self.conv1(x)
        return x

tudui = Tudui()
torch.save(tudui, "tudui_method1.pth")
