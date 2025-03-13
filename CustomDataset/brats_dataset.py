import torch
import torchvision
import torch.nn as nn
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
import nibabel as nib

class BratsDataset(Dataset):
    """" My custom dataset used to load the brats2020 dataset """

    def __init__(self, transforms=None):
        self.root = "dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/"
        self.csv = self.root + "name_mapping.csv"

        # List of all sample names
        self.name_mapping = list(pd.read_csv(self.csv)["BraTS_2020_subject_ID"])
        self.transforms = transforms

        self.file_names_suffix = ["_flair", "_t1", "_t1ce", "_t2"]

        # Maps channel dim to name
        self.dim_mapping = {i:string.removeprefix("_") for i, string in enumerate(self.file_names_suffix)}
    

    def load_sample_input(self, index):
        """ Loads sample n, stacks different images as channels """

        imgs = []
        for suffix in self.file_names_suffix:
            img = nib.load(f"{self.root}/{self.name_mapping[index]}/{self.name_mapping[index]}{suffix}.nii")
            img_numpy = img.get_fdata()
            img_numpy = img_numpy/img_numpy.max()
            imgs.append(img_numpy)

        stacked = np.stack(imgs, -1)
        return torch.from_numpy(stacked).float()


    def load_sample_seg(self, index):
        seg = nib.load(f"{self.root}/{self.name_mapping[index]}/{self.name_mapping[index]}_seg.nii").get_fdata()
        return torch.from_numpy(seg).float()


    def __len__(self):
        return len(self.name_mapping)
    
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        stacked_input = self.load_sample_input(idx)
        segmentation = self.load_sample_seg(idx)

        if self.transforms:
            stacked_input = self.transforms(stacked_input)
        return (stacked_input, segmentation)



dataset = BratsDataset()

