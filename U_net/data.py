
from torch.utils.data import Dataset
import os
from utils import *
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])

class MyDataset(Dataset):
    def __init__(self,path):
        self.path = path # 保存根目录
        # 拼接得到'./data/SegmentationClass
        # 列出上面这个路径下的所有文件名，保存为列表
        self.name = os.listdir(os.path.join(path,'SegmentationClass'))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        segment_name = self.name[index]# xx.png
        # ./data/SegmentationClass/cat.png
        segment_path = os.path.join(self.path,'SegmentationClass',segment_name)
        # 找到原始图片的路径
        image_path = os.path.join(self.path,'JPEGImages',segment_name.replace('png','jpg'))

        # 对图片mask统一大小
        segment_image = keep_image_size_open(segment_path)
        image = keep_image_size_open(image_path)
        return  transform(image),transform(segment_image)

if __name__ == '__main__':
    data = MyDataset('data/VOC/VOC2012_train_val')
    print(data[0][0].shape)
    print(data[0][1].shape)
