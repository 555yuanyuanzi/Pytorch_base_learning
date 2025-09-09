from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("dataset/train/ants_image/0013035.jpg")
print(img)

# Totensor的使用
trans_totensor = transforms.ToTensor()# 创建totensor对象
img_tensor = trans_totensor(img)
writer.add_image("ToTensor",img_tensor)


# Normalize归一化
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize",img_norm)

# Resize
print(img.size)
trans_resize = transforms.Resize((512,512))
# img PIL->resize->img_resize PIL
img_resize = trans_resize(img)
# img_resize PIL->totensor->img_resize tensor
img_resize = trans_totensor(img_resize)
writer.add_image("Resize",img_resize,0)
print(img_resize)

# Compose--resize-2
trans_resize_2 = transforms.Resize(512)# 只输入一个值：最小边和512匹配
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])# 参数列表：改变图像大小，转换类型
img_resize_2 = trans_compose(img)
writer.add_image("Resize",img_resize_2,1)

# RandomCrop随机裁剪
trans_random = transforms.RandomCrop((500,100))
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)

writer.close()