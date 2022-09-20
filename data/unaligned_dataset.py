import os
import json
import random
from glob import glob

import albumentations as A
import cv2
import numpy as np
import torch
import torch.utils.data as data
from albumentations.pytorch import ToTensorV2


class UnAlignedDataset(data.Dataset):
    def __init__(self, dataset_dir, img_size=(256, 512), phase='train'):
        super().__init__()
        self.img_size = img_size    
        self.dataset_dir = dataset_dir
        self.A_names = sorted(os.listdir(os.path.join(self.dataset_dir, f"{phase}_A")))
        self.B_names = sorted(os.listdir(os.path.join(self.dataset_dir, f"{phase}_B")))
        self.back_paths = glob(os.path.join(self.dataset_dir, "background", '*'))
        self.phase = phase
        with open(os.path.join(self.dataset_dir, "out_ang.json"), "r") as f:
            self.out_ang = json.load(f)
        with open(os.path.join(self.dataset_dir, "ang_studio.json"), "r") as f:
            self.ang_studio = json.load(f)
        
        self.hflip = A.Compose([
            A.HorizontalFlip(),
        ], additional_targets={'image0': 'image'})

        self.aug_transform = A.Compose([
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), rotate=(-5, 5), shear=(-5, 5), p=0.8),
            A.Resize(*self.img_size),
            # A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            # ToTensorV2(),
        ], additional_targets={'image0': 'image'})

        self.back_transform = A.Compose([
            A.Flip(),
            A.GridDistortion(distort_limit=0.5, p=0.6),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=100, approximate=True, same_dxdy=True, p=0.6),
            A.RandomResizedCrop(*self.img_size, scale=(0.6, 1.0), ratio=(0.6, 2.)),
            A.ColorJitter(brightness=(0.8, 1.1), contrast=0.4, saturation=0.4, hue=0.05),
            A.Affine(scale=(0.8, 1.2), translate_percent=(-0.25, 0.25), rotate=(-180, 180), shear=(-25, 25), interpolation=cv2.INTER_LINEAR, p=0.7, mode=cv2.BORDER_REFLECT),
            A.Resize(*self.img_size, interpolation=cv2.INTER_LINEAR),
            # A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            # ToTensorV2(),
        ])


    def overlay_refl(self, img):
        back_path = self.back_paths[random.randint(0, len(self.back_paths) - 1)]
        back_arr = cv2.imread(back_path, cv2.IMREAD_COLOR)
        back_arr = cv2.cvtColor(back_arr, cv2.COLOR_BGR2RGB)
        back_arr = self.back_transform(image=back_arr)['image']
        mask = img[:,:,3:4]/255
        img = img[:,:,:3]
        back_arr = np.clip(back_arr, 50, 255)
        back_arr = back_arr * mask
        alpha = np.clip(random.uniform(0., 1.1), 0, 1)
        out_img = img*(1-alpha) + back_arr*alpha
        return out_img

    def __getitem__(self, index):
        A_file = self.A_names[index]
        ang = self.out_ang.get(A_file, 0)
        A = cv2.imread(os.path.join(self.dataset_dir, f"{self.phase}_A", A_file), cv2.IMREAD_UNCHANGED)
        B_names = self.ang_studio[str(ang)]
        B_file = B_names[random.randint(0, len(B_names) - 1)]
        if B_file not in self.B_names:
            B_file = self.B_names[random.randint(0, len(self.B_names) - 1)]
        B = cv2.imread(os.path.join(self.dataset_dir, f"{self.phase}_B", B_file), cv2.IMREAD_UNCHANGED)
        A = cv2.cvtColor(A, cv2.COLOR_BGRA2RGBA)
        B = cv2.cvtColor(B, cv2.COLOR_BGRA2RGB)

        aug = self.hflip(image=A, image0=B)
        A = aug['image']
        B = aug['image0']
        A = self.aug_transform(image=A)['image']
        B = self.aug_transform(image=B)['image']
        if random.uniform(0,1) < 0.7:
            if random.uniform(0,1) < 0.1:
                A = B
            A = self.overlay_refl(A)
        A = A[:,:,:3]
        A = torch.from_numpy(A/127.5 - 1).permute(2,0,1).float()
        B = torch.from_numpy(B/127.5 - 1).permute(2,0,1).float()
        return A, B

    def __len__(self):
        return len(self.A_names)
