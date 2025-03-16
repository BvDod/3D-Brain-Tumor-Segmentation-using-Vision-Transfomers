# %%
import torch

from functions.transforms import get_transforms_3d_val
from CustomDataset.brats_dataset import BratsDataset
from models.vit import VIT
from torch.utils.data import DataLoader
from functions import visualize

import numpy as np

from PIL import Image as im
from pathlib import Path

torch.manual_seed(0)

settings = {
    "dataset": "MNIST",

    "print_debug": False,
    "batch_size": 2,
    "learning_rate": 1e-4, # for Mnsist
    "max_epochs": 100,
    "early_stopping_epochs": 50,

    "model_settings" : {
        "patch_size": 32,
        "embedding_size": 256,
        "attention_heads": 8,
        "transformer_layers": 8
    }
    }

 # Print settings and info
device = "cpu" if torch.cuda.is_available() else "cpu"
print(str(settings))
print(f"Device: {device}" + "\n")

# Loading dataset
transforms = get_transforms_3d_val(settings["model_settings"]["patch_size"])

dataset = BratsDataset(transforms=transforms, device=device)
dataset_train, dataset_val = torch.utils.data.random_split(dataset, [0.8, 0.2], )

dataloader_test = DataLoader(dataset_val, batch_size=settings["batch_size"], pin_memory=True, num_workers = 0,)

train_sample = dataset_val[0]
input_shape, mask_shape = train_sample["image"].shape, train_sample["label"].shape
print(input_shape, mask_shape)

# Setting up model
model_settings = settings["model_settings"]
model_settings["num_channels"] = input_shape[0]
model_settings["input_shape"] = input_shape
model_settings["device"] = device

model = VIT(model_settings)
model.to(device)
model.load_state_dict(torch.load("models/saved_models/model_latest.pt", weights_only=True))


model.eval() 

# %%
from monai.metrics import DiceMetric

dice_metric = DiceMetric(include_background=True)
model.eval()
with torch.no_grad(): 
    for train_sample in dataloader_test:
        x_test, y_test = train_sample["image"], train_sample["label"],
        x_test, y_test = x_test.to(device), y_test.to(device)  
        
        res = model(x_test)
        pred = torch.nn.functional.one_hot(res.argmax(dim=1), num_classes=5).movedim(-1,1)
        dice_metric(y_pred=pred, y=y_test)
    metric = dice_metric.aggregate().item()
    print(f"Mean Dice score: {metric}")

    visualize.create_segmentation_png_seq(x_test[0], pred[0], "prediction0/", dim=0)
    visualize.create_segmentation_png_seq(x_test[0], y_test[0], "ground_truth0/", dim=0)
    visualize.create_segmentation_png_seq(x_test[0], pred[0], "prediction1/", dim=1)
    visualize.create_segmentation_png_seq(x_test[0], y_test[0], "ground_truth1/", dim=1)
    visualize.create_segmentation_png_seq(x_test[0], pred[0], "prediction2/", dim=2)
    visualize.create_segmentation_png_seq(x_test[0], y_test[0], "ground_truth2/", dim=2)
    """
    x_test = x_test.movedim(-1,1)
    images = visualize.add_segmentation_to_image(x_test[0], pred[0])
    images = np.concatenate(images, axis=2)
    array = images.astype(np.uint8)
    array = np.moveaxis(array, 0,-1)
    image = im.fromarray(array)
    foldername = "prediction/"
    Path(foldername).mkdir(parents=True, exist_ok=True)
    image.save(f"{foldername}output.png")

    images = visualize.add_segmentation_to_image(x_test[0], y_test[0])
    images = np.concatenate(images, axis=2)
    array = images.astype(np.uint8)
    array = np.moveaxis(array, 0,-1)
    image = im.fromarray(array)
    foldername = "ground_truth/"
    Path(foldername).mkdir(parents=True, exist_ok=True)
    image.save(f"{foldername}output.png")
    """

