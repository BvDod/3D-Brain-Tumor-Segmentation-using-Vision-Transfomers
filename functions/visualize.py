from CustomDataset.brats_dataset import BratsDataset
from skimage import io, color
from skimage.transform import rotate
import matplotlib.pyplot as plt
import numpy as np
import random
import imageio
from pathlib import Path

from PIL import Image as im 


def add_segmentation_to_image(x, y, x_channel=0, dim=None):
    """
    displays seg (y) on top of image input (x)
    x_channel: which channel to use for the image
    dim: which dimension to display the segmentation on. If None, all orientations are returned
    """

    x = x.movedim(0, -1)
    x, y = x.cpu().numpy(), y.cpu().numpy()
    x = x[:,:,:,x_channel].squeeze() # Select channel and make 2d
    y = y.squeeze()
    shape = x.shape
    
    images = []
    colors = [(125,125,0),(0,0,255),(0,255,0), (255,0,0)] # Colors for each class
    
    # If make seg images in all orientations
    if not dim:
        for dim in range(0,3):
            
            x_slice = np.take(x, indices = shape[dim]//2, axis=dim)
            y_slice = np.take(y, indices = shape[dim]//2, axis=dim)
            x_slice = color.gray2rgb(x_slice)

            colors_slice = select_color_subsection_labels(y_slice, colors)
            x_slice = x_slice * 255
            image = color.label2rgb(y_slice.astype(int),image=x_slice,colors=colors_slice,alpha=0.75, bg_label=0, bg_color=None)
            np.moveaxis(image, -1, 0)
            images.append(image)
    
    # Orientation is specified
    else:
        x_slice = np.take(x, indices = shape[dim]//2, axis=dim)
        y_slice = np.take(y, indices = shape[dim]//2, axis=dim)

        colors_slice = select_color_subsection_labels(y_slice, colors)
        image = color.label2rgb(y_slice.astype(int),image=x_slice,colors=colors_slice, alpha=0.75, bg_label=0, bg_color=None)
        images.append(image)
    
    # Pad all images to the same size
    x_max = max(img.shape[0] for img in images)
    y_max = max(img.shape[1] for img in images)
    images_padded = []
    for image in images:
        x, y, _ = image.shape
        x_pad = (x_max - x) //2
        y_pad = (y_max - y) //2
        image_padded = np.pad(image,pad_width=[(x_pad, x_pad), (y_pad, y_pad), (0,0)] ,mode="constant", constant_values=0)
        image_padded = np.moveaxis(image_padded, -1, 0)
        images_padded.append(image_padded)
        
    # Return batched images
    return np.stack(images_padded)


def select_color_subsection_labels(labels, colors):
    """
    determines which colors to keep based on their presence in the labels
    """
    colors_slice = []
    unique_labels = list(np.unique(labels.astype(int)))
    for label in unique_labels:
        if label == 0:
            continue
        colors_slice.append(colors[label-1])
    return colors_slice


def create_segmentation_png_seq(x, y, foldername, x_channel=1, dim=1):
    """
    Saves sequence of PNG's with overlayed segmentation
    """

    x, y = x.numpy(), y.numpy()
    x = x[:,:,:,x_channel].squeeze()

    shape = x.shape

    for i in range(shape[dim]):
        x_slice = color.gray2rgb(np.take(x,indices=i, axis=dim))
        y_slice = np.take(y, indices = i, axis=dim)

        colors = [(125,125,0),(0,0,255),(0,255,0), (255,0,0)]
        colors_slice = select_color_subsection_labels(y_slice, colors)

        x_slice = x_slice * 255
        image = color.label2rgb(y_slice.astype(int),image=x_slice,colors=colors_slice,alpha=0.75, bg_label=0, bg_color=None)

        array = (image).astype(np.uint8)
        image = im.fromarray(array)

        Path(foldername).mkdir(parents=True, exist_ok=True)
        image.save(f"{foldername}{i}{dim}{x_channel}.png")

