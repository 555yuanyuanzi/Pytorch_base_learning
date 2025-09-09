import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

dataset_transform = transforms.Compose([
    transforms.ToTensor(),
])
tran_set = torchvision.datasets.CIFAR10(root="./dataset", transform = dataset_transform, train=True ,download= False)
test_set = torchvision.datasets.CIFAR10(root="./dataset", transform = dataset_transform, train=False ,download= False)

# print(test_set[0])
# print(test_set.classes)
#
# img,target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()

# print(test_set[0])

writer = SummaryWriter("p10")
for i in range(10):
    img,target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()