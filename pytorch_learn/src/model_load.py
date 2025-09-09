import torch
import torchvision
# import model_save
from torch import nn

# 方式一->保存方式1加载模型
# model = torch.load("vgg16_method1.pth")
# print(model)

# 方式2->加载模型
vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))

# model = torch.load("vgg16_method2.pth")
# print(vgg16)

#陷阱1
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(3,64,3)

    def forward(self, x):
        x = self.conv1(x)
        return x

model = torch.load("tudui_method1.pth")
print(model)