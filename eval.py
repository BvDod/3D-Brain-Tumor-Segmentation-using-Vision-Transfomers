import torch
import numpy as np

from monai.metrics import DiceMetric

from functions.transforms import get_transforms_3d_val
from CustomDataset.brats_dataset import BratsDataset
from models.vit3d import VIT3Dsegmentation
from torch.utils.data import DataLoader

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
    }}

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

model = VIT3Dsegmentation(model_settings)
model.to(device)
model.load_state_dict(torch.load("models/saved_models/model_latest.pt", weights_only=True))


# Evaluate Model
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