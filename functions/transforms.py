import monai
from monai.transforms import RandFlipd, NormalizeIntensityd, NormalizeIntensityd, RandScaleIntensityd, RandShiftIntensityd, RandSpatialCropd
from torchvision import transforms

def get_transforms_3d(patch_size):
    transforms = monai.transforms.Compose([
        RandSpatialCropd(keys=["image", "label"], roi_size=(192, 192, 128), random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        monai.transforms.DivisiblePadd(keys=["image", "label"], k=patch_size),
    ])
    return transforms