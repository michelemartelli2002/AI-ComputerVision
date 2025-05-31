import os
from PIL import Image
from torch.utils.data import Dataset
import sys

class CatBreeds67(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform

        self.objs = []
        self.classes = []

        for idx, class_name in enumerate(sorted(os.listdir(root))):
            class_path = os.path.join(root, class_name)
            if os.path.isdir(class_path):
                self.classes.append(class_name)
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.objs.append([img_path, idx])

    def __len__(self):
        return len(self.objs)

    def __getitem__(self, idx):
        elem = self.objs[idx]
        image = Image.open(elem[0]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if not(elem[1] >= 0 and elem[1] < 67):
            sys.__stdout__.write(f"{elem=}\n")
            sys.__stdout__.flush()  # ensure it appears immediately

        return image, elem[1]
