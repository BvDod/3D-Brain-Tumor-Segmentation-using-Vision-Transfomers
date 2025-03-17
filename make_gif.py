# Description: This script is used to create pngs of the segmentation of the model, which can be turned into a gif (i use ezgif)
# Also outputs static images of the segmenation from all 3 orientations

import torch

from functions.transforms import get_transforms_3d_val
from CustomDataset.brats_dataset import BratsDataset
from models.vit3d import VIT3Dsegmentation
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

dataloader_train = DataLoader(dataset_train, batch_size=settings["batch_size"], shuffle=True, drop_last=True, pin_memory=False, num_workers = 6)
dataloader_test = DataLoader(dataset_val, batch_size=settings["batch_size"], pin_memory=False, num_workers = 6)

train_sample = dataset_train[0]
input_shape, mask_shape = train_sample["image"].shape, train_sample["label"].shape

# Setting up model
model_settings = settings["model_settings"]
model_settings["num_channels"] = input_shape[0]
model_settings["input_shape"] = input_shape
model_settings["device"] = device

model = VIT3Dsegmentation(model_settings)
model.to(device)

# Load model to use
model.load_state_dict(torch.load("models/saved_models/model_latest.pt", weights_only=True))


model.eval() 
with torch.no_grad():    
    train_sample = dataset_val[8]
    x_test, y_test = train_sample["image"].unsqueeze(0), train_sample["label"].unsqueeze(0),

    x_test, y_test = x_test.to(device), y_test.to(device)  
    res = model(x_test)

    pred = res.detach().movedim(1,-1).argmax(dim=-1)
    x_test = x_test.movedim(1,-1)
    y_test = y_test.argmax(dim=1)
    
    # Create gifs for every orientation, for both the prediction and the ground truth
    visualize.create_segmentation_png_seq(x_test[0], pred[0], "prediction0/", dim=0)
    visualize.create_segmentation_png_seq(x_test[0], y_test[0], "ground_truth0/", dim=0)
    visualize.create_segmentation_png_seq(x_test[0], pred[0], "prediction1/", dim=1)
    visualize.create_segmentation_png_seq(x_test[0], y_test[0], "ground_truth1/", dim=1)
    visualize.create_segmentation_png_seq(x_test[0], pred[0], "prediction2/", dim=2)
    visualize.create_segmentation_png_seq(x_test[0], y_test[0], "ground_truth2/", dim=2)
    
    # Create a single image for the prediction, with all 3 orientations
    x_test = x_test.movedim(-1,1)
    images = visualize.add_segmentation_to_image(x_test[0], pred[0])
    images = np.concatenate(images, axis=2)
    array = images.astype(np.uint8)
    array = np.moveaxis(array, 0,-1)
    image = im.fromarray(array)
    foldername = "prediction/"
    Path(foldername).mkdir(parents=True, exist_ok=True)
    image.save(f"{foldername}output.png")

    # Create a single image for the ground truth, with all 3 orientations
    images = visualize.add_segmentation_to_image(x_test[0], y_test[0])
    images = np.concatenate(images, axis=2)
    array = images.astype(np.uint8)
    array = np.moveaxis(array, 0,-1)
    image = im.fromarray(array)
    foldername = "ground_truth/"
    Path(foldername).mkdir(parents=True, exist_ok=True)
    image.save(f"{foldername}output.png")

