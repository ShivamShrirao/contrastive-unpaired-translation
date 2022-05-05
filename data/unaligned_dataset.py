import os
import cv2
# from glob import glob
# import numpy as np
# from PIL import Image

import random
import torch.utils.data as data
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform


class UnAlignedDataset(data.Dataset):
    def __init__(self, dataset_dir, img_size, phase='train'):
        super().__init__()
        self.img_size = img_size
        self.dataset_dir = dataset_dir
        self.A_names = sorted(os.listdir(os.path.join(self.dataset_dir, f"{phase}_A")))
        self.B_names = sorted(os.listdir(os.path.join(self.dataset_dir, f"{phase}_B")))
        self.phase = phase

        if phase == 'test':
            self.aug_transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
                ToTensorV2(),
            ], additional_targets={'image0': 'image'})
        else:
            self.aug_transform = A.Compose([
                A.HorizontalFlip(),
                A.Affine(scale=(0.7, 1.2), translate_percent=(-0.15, 0.15), rotate=(-20, 20),
                        shear=(-10, 10), p=0.7),
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
                ToTensorV2(),
            ], additional_targets={'image0': 'image'})

    def __getitem__(self, index):
        A_path = self.A_names[index]
        A = cv2.imread(os.path.join(self.dataset_dir, f"{self.phase}_A", A_path))
        B_path = self.B_names[random.randint(0, len(self.B_names) - 1)]
        B = cv2.imread(os.path.join(self.dataset_dir, f"{self.phase}_B", B_path))
        A = cv2.cvtColor(A, cv2.COLOR_BGR2RGB)
        B = cv2.cvtColor(B, cv2.COLOR_BGR2RGB)
        data = self.aug_transform(image=A, image0=B)
        A, B = data['image'], data['image0']
        return A, B

    def __len__(self):
        return len(self.A_names)
