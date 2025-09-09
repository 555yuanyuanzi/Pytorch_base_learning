import torch
import torchvision
from PIL import Image
from torch import nn

image_path = "imgs/dog3.png"
image = Image.open(image_path)
image = image.convert('RGB')
print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                           torchvision.transforms.ToTensor()])
image = transform(image)
print(image.shape)

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

# 方式1保存怎么加载？
model = torch.load("tudui_gpu_29.pth")
print(model)
# 方式2保存怎么加载？
# # --加载网络模型
# # 实例化你的模型结构
# model = Tudui()
# model = model.cuda()
# # 加载状态字典（权重）
# checkpoint = torch.load("tudui_gpu_29.pth")
# # 将加载的权重加载到模型实例中
# model.load_state_dict(checkpoint)
# print(model)

image = torch.reshape(image,(1,3,32,32))
image = image.cuda()
model.eval()
with torch.no_grad():
    output = model(image)

print(output)
print(output.argmax(1))