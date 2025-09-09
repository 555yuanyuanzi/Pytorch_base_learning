import os.path

import torch
import  numpy as np
from torchvision.utils import save_image

from data import transform
from net import *
from train import weight_path
from utils import keep_image_size_open

net = UNet().cuda()

weights = 'params/unet.pth'
if os.path.exists(weights):
    net.load_state_dict(torch.load(weights))
    print('成功')
else:
    print('没有加载权重')

_input = input('请输入图片的路径：')

img = keep_image_size_open(_input)
img_data = transform(img).cuda()
print(img_data.shape)
# 升维
img_data = torch.unsqueeze(img_data,dim=0)
out = net(img_data)
save_image(out, 'result/result.jpg')
print(out)


