from torch.utils.data import Dataset
import cv2
from PIL import Image
import os

class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)


    def __getitem__(self , idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir.split('_')[0]
        return img, label

    def __len__(self):
        return len(self.img_path)

root_dir = "dataset/train"
ants_label_dir = "ants_image"
bees_label_dir = "bees_image"
ants_dataset = MyData(root_dir , ants_label_dir)
bees_dataset = MyData(root_dir , bees_label_dir)

train_dataset = ants_dataset + bees_dataset

# 标签文件生成逻辑
target_dir = 'ants_image'
target_path = os.path.join(root_dir, target_dir)
img_path = os.listdir(target_path)
label = target_dir.split('_')[0]
out_dir = 'ants_label'
for img_name in img_path:
    # 正确分割文件名和扩展名
    file_name = os.path.splitext(img_name)[0]
    label_file = os.path.join(root_dir, out_dir, f"{file_name}.txt")
    with open( label_file, 'w+') as f:
        f.write(label)