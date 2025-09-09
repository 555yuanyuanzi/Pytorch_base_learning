from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision

# 准备测试数据集
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

# 测试数据集中第一张图片及targettensor
img, target = test_data[0]
print(img.shape)
print(target)


writer = SummaryWriter("dataloader")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(img.shape)
        # print(targets)
        writer.add_images("Epoch:{}".format(epoch), imgs, step)
        step = step + 1

writer.close()