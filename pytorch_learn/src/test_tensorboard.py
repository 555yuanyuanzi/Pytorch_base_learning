from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
# 向 TensorBoard 中添加标量数据（如损失值、准确率等随步数变化的数值）
writer = SummaryWriter("logs")
image_path = "dataset/train/ants_image/28847243_e79fe052cd.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(img_array.shape)

# 需要指定shape的格式,用于向 TensorBoard 中添加图像数据
writer.add_image("train", img_array, 2, dataformats="HWC")

for i in range(100):
     writer.add_scalar("y=3x", 3*i , i)

writer.close()