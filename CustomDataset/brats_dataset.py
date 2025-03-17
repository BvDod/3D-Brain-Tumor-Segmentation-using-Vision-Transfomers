import torch
import pandas as pd

from torch.utils.data import Dataset
import nibabel as nib


class BratsDataset(Dataset):
    """" My custom dataset used to load the brats2020 dataset """


    def __init__(self, transforms = None, device="cuda"):
        self.root = "dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/"
        self.csv = self.root + "name_mapping.csv"
        self.device = device

        # List of all sample names
        self.name_mapping = list(pd.read_csv(self.csv)["BraTS_2020_subject_ID"])
        self.transforms = transforms

        self.file_names_suffix = ["_flair", "_t1", "_t1ce", "_t2"]

        # Maps channel dim to name
        self.dim_mapping = {i:string.removeprefix("_") for i, string in enumerate(self.file_names_suffix)}
    

    def load_sample_input(self, index):
        """ Loads sample image n, stacks different images as channels """

        imgs = []
        for suffix in self.file_names_suffix:
            img = nib.load(f"{self.root}/{self.name_mapping[index]}/{self.name_mapping[index]}{suffix}.nii")
            img_tensor = torch.from_numpy(img.get_fdata())
            img_tensor = img_tensor/img_tensor.max()
            imgs.append(img_tensor)    
        stacked = torch.stack(imgs, -1).float()
        return stacked


    def load_sample_seg(self, index):
        """ loads sample segmentation n """
        seg = nib.load(f"{self.root}/{self.name_mapping[index]}/{self.name_mapping[index]}_seg.nii").get_fdata()
        seg = torch.from_numpy(seg).long()
        return seg


    def __len__(self):
        return len(self.name_mapping)
    
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        stacked_input = self.load_sample_input(idx)
        segmentation = self.load_sample_seg(idx)

        # convert int label to onehot
        segmentation = torch.nn.functional.one_hot(segmentation, num_classes=5)
        
        stacked_input = stacked_input.movedim(-1, 0)
        segmentation = segmentation.movedim(-1, 0)

        sample = {"image": stacked_input, "label": segmentation}
        
        if self.transforms:
            return self.transforms(sample)
        return sample



dataset = BratsDataset()

