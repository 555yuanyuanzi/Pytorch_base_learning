from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path = "dataset/train/ants_image/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")

# 怎么使用transforms
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
# tensor包含了神经网络相关理论基础参数

writer.add_image("Tensor_img", tensor_img)
writer.close()

