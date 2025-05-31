from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
import sys
import random


class Caltech101(Dataset):
    def __init__(self, transform=None, split='train'):
        assert split in ['train', 'val', 'test']
        root = "data/caltech-101/101_ObjectCategories"
        self.transform = transform
        self.split = split
        self.classes = []
        self.data = []
        split_ratio = [0.8, 0.1, 0.1]

        for idx, class_name in enumerate(sorted(os.listdir(root))):
            class_path = os.path.join(root, class_name)
            if not os.path.isdir(class_path):
                continue

            self.classes.append(class_name)
            images = [os.path.join(class_path, f)
                      for f in os.listdir(class_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            images.sort()
            random.shuffle(images)

            n = len(images)
            n_train = int(split_ratio[0] * n)
            n_val = int(split_ratio[1] * n)

            if split == 'train':
                selected = images[:n_train]
            elif split == 'val':
                selected = images[n_train:n_train + n_val]
            else:
                selected = images[n_train + n_val:]

            self.data.extend((img_path, idx) for img_path in selected)

        self.num_classes = len(self.classes)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label