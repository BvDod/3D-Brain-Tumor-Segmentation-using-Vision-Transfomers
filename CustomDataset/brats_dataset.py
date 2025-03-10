import torch
import torchvision
import torch.nn as nn
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
import nibabel as nib

class BratsDataset(Dataset):

    def __init__(self, transforms=None):
        self.root = "dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/"
        self.csv = self.root + "name_mapping.csv"

        self.name_mapping = list(pd.read_csv(self.csv)["BraTS_2020_subject_ID"])
        self.transforms = transforms

        self.file_names_suffix = ["_flair", "_t1", "_t1ce", "_t2"]
    
    def __len__(self):
        return len(self.name_mapping)
    
    def __getitem__(self, idx):
        imgs = []
        for suffix in self.file_names_suffix:
            img = nib.load(f"{self.root}/{self.name_mapping[idx]}/{self.name_mapping[idx]}{suffix}.nii")
            imgs.append(img.get_fdata())
        stacked = np.stack(imgs, -1)
        seg = img = nib.load(f"{self.root}/{self.name_mapping[idx]}/{self.name_mapping[idx]}_seg.nii")
        print(seg.shape)
        print(stacked.shape)



dataset = BratsDataset()
dataset[0]