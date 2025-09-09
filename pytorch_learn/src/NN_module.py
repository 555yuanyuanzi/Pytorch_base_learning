import torch
from torch import nn

class Tudui(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,input):
        output = input + 1
        return output

tudui = Tudui()
x = torch.tensor(1.0)
output = tudui(x)# PyTorch 的nn.Module类实现了__call__方法，它会在内部调用forward方法
print(output)