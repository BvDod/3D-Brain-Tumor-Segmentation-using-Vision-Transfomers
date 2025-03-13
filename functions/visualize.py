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
    """
    x, y = x.cpu().numpy(), y.cpu().numpy()
    x = x[:,:,:,x_channel].squeeze() # Select channel and make 2d

    shape = x.shape
    images = []

    colors = [(255,0,0),(0,0,255),(0,255,0), (0,125,125)]
    # If make seg images in all orientations
    if not dim:
        for dim in range(0,3):
            x_slice = np.take(x, indices = shape[dim]//2, axis=dim)
            y_slice = np.take(y, indices = shape[dim]//2, axis=dim)
            
            colors_slice = select_color_subsection_labels(y_slice, colors)
            image = color.label2rgb(y_slice.astype(int),image=x_slice,colors=colors_slice,alpha=0.1, bg_label=0, bg_color=None)
            images.append(image)
    
    # Orientation is specified
    else:
        x_slice = np.take(x, indices = shape[dim]//2, axis=dim)
        y_slice = np.take(y, indices = shape[dim]//2, axis=dim)

        colors_slice = select_color_subsection_labels(y_slice, colors)
        image = color.label2rgb(y_slice.astype(int),image=x_slice,colors=colors_slice, alpha=0.1, bg_label=0, bg_color=None)
        images.append(image)
    return images


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

        colors = [(255,0,0),(0,0,255),(0,255,0), (0,125,125)]
        colors_slice = select_color_subsection_labels(y_slice, colors)

        image = color.label2rgb(y_slice.astype(int),image=x_slice,colors=colors_slice,alpha=0.1, bg_label=0, bg_color=None)

        array = (image*255).astype(np.uint8)
        image = im.fromarray(array)

        Path(foldername).mkdir(parents=True, exist_ok=True)
        image.save(f"{foldername}{i}{dim}{x_channel}.png")



if __name__ == "__main__":
    dataset = BratsDataset()
    random_sample = random.randint(0,len(dataset)-1)
    print(f"Random index: {random_sample}")
    x, y = dataset[random_sample]
    print(np.unique(x))
    create_segmentation_png_seq(x, y, "test/001/", x_channel=1, dim=1,)
    images = add_segmentation_to_image(x, y)
    for image in images:
        print(np.unique(image.astype(np.uint8)))
        plt.imshow(image)
        plt.show()


