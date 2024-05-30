import torch 
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import os
import numpy as np 
from PIL import Image
# 创建自定义的 Dataset 类
class CUBDataset(Dataset):
    def __init__(self, root_dir, image_paths, image_labels, image_splits, train=True, transform=None):
        self.root_dir = root_dir
        self.image_paths = image_paths
        self.image_labels = image_labels
        self.image_splits = image_splits
        self.train = train
        self.transform = transform

        # 获取所需的图片ID
        self.image_ids = [img_id for img_id, is_train in image_splits.items() if is_train == train]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.root_dir, 'images', self.image_paths[img_id])
        image = Image.open(img_path).convert('RGB')
        label = self.image_labels[img_id]

        if self.transform:
            image = self.transform(image)

        return image, label
    
# 定义数据增强和标准化
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}