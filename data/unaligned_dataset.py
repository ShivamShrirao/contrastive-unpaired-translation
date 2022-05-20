import os
import random
from glob import glob

import albumentations as A
import cv2
import numpy as np
import torch
import torch.utils.data as data
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2


class UnAlignedDataset(data.Dataset):
    def __init__(self, dataset_dir, img_size=(256, 512), phase='train'):
        super().__init__()
        self.img_size = img_size
        self.dataset_dir = dataset_dir
        self.A_names = sorted(os.listdir(os.path.join(self.dataset_dir, f"{phase}A")))
        self.B_names = sorted(os.listdir(os.path.join(self.dataset_dir, f"{phase}B")))
        self.back_paths = glob(os.path.join(self.dataset_dir, "background", '*'))
        self.phase = phase

        self.aug_transform = A.Compose([
            A.HorizontalFlip(),
            A.Resize(*self.img_size),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ], additional_targets={'image0': 'image'})

        self.back_transform = A.Compose([
            A.Flip(),
            A.GridDistortion(distort_limit=0.6, p=0.6),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=150, approximate=True, same_dxdy=True, p=0.6),
            A.RandomResizedCrop(*self.img_size, scale=(0.5, 1.0), ratio=(0.5, 2.)),
            A.RandomSunFlare(num_flare_circles_lower=1, num_flare_circles_upper=6, src_radius=100, p=0.4),
            FadeEdges(always_apply=True),
            A.ColorJitter(brightness=(0.5, 1.1), contrast=0.4, saturation=0.4, hue=0.05),
            A.CoarseDropout(max_holes=4, max_height=50, max_width=50, p=0.25),
            A.Affine(scale=(0.7, 1.), translate_percent=(-0.25, 0.25), rotate=(-180, 180), shear=(-25, 25), interpolation=cv2.INTER_LINEAR, p=0.7),
            A.Resize(*self.img_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])


    def overlay_refl(self, img):
        back_path = self.back_paths[random.randint(0, len(self.back_paths) - 1)]
        back_arr = cv2.imread(back_path, cv2.IMREAD_COLOR)
        back_arr = cv2.cvtColor(back_arr, cv2.COLOR_BGR2RGB)
        back_arr = self.back_transform(image=back_arr)['image']
        alpha = random.uniform(0.5, 1)
        out_img = img*alpha + back_arr*(1-alpha)
        return out_img

    def __getitem__(self, index):
        A_path = self.A_names[index]
        A = cv2.imread(os.path.join(self.dataset_dir, f"{self.phase}A", A_path))
        B_path = self.B_names[random.randint(0, len(self.B_names) - 1)]
        B = cv2.imread(os.path.join(self.dataset_dir, f"{self.phase}B", B_path))
        A = cv2.cvtColor(A, cv2.COLOR_BGR2RGB)
        B = cv2.cvtColor(B, cv2.COLOR_BGR2RGB)
        data = self.aug_transform(image=A, image0=B)
        A, B = data['image'], data['image0']
        if random.uniform(0,1) < 0.3:
            A = self.overlay_refl(A)
        return A, B

    def __len__(self):
        return len(self.A_names)


class FadeEdges(ImageOnlyTransform):
    def apply(self, image, **kwargs):
        h, w, c = image.shape
        diam = np.random.randint(2, 13)
        frame = np.zeros((h // 4, w // 4), dtype=np.float32)
        frame[diam:-diam, diam:-diam].fill(1.)
        ksz = 2 * diam + 1
        frame = cv2.GaussianBlur(frame, (ksz, ksz), 0.3 * ((ksz - 1) * 0.5 - 1) + 0.8)
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.uint8(image * frame[..., None])
