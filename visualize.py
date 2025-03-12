from CustomDataset.brats_dataset import BratsDataset
from skimage import io, color
from skimage.transform import rotate
import matplotlib.pyplot as plt
import numpy as np
import random

from PIL import Image as im 

def add_segmentation_to_image(x, y, x_channel=0, dim=None):
    """
    displays seg (y) on top of image input (x)
    """
    x, y = x.numpy(), y.numpy()
    x = x[:,:,:,x_channel].squeeze() # Select channel and make 2d

    shape = x.shape
    images = []

    # If make seg images in all orientations
    if not dim:
        for dim in range(0,3):
            x_slice = np.take(x, indices = shape[dim]//2, axis=dim)
            y_slice = np.take(y, indices = shape[dim]//2, axis=dim)
            image = color.label2rgb(y_slice.astype(int),image=x_slice,colors=[(255,0,0),(0,0,255),(0,255,0)],alpha=0.1, bg_label=0, bg_color=None)
            images.append(image)
    
    # Orientation is specified
    else:
        x_slice = np.take(x, indices = shape[dim]//2, axis=dim)

        y_slice = np.take(y, indices = shape[dim]//2, axis=dim)
        image = color.label2rgb(y_slice.astype(int),image=x_slice,colors=[(255,0,0),(0,0,255),(0,255,0)],alpha=0.1, bg_label=0, bg_color=None)
        images.append(image)
    return images

def create_segmentation_gif(x, y, x_channel=0, dim=0):
    x, y = x.numpy(), y.numpy()
    x = x[:,:,:,x_channel].squeeze()

    shape = x.shape
    images = []

    for i in range(shape[dim]):
        x_slice = color.gray2rgb(np.take(x, indices = i, axis=dim))
        y_slice = np.take(y, indices = i, axis=dim)
        image = color.label2rgb(y_slice.astype(int),image=x_slice,colors=[(255,0,0),(0,0,255),(0,255,0)],alpha=0.1, bg_label=0, bg_color=None)
        images.append(im.fromarray((image*10).astype(np.uint8)))
    images[0].save("out.gif", save_all=True, append_images=images[1:], duration=10)



if __name__ == "__main__":
    dataset = BratsDataset()
    random_sample = random.randint(0,len(dataset)-1)
    print(f"Random index: {random_sample}")
    x, y = dataset[random_sample]
    print(np.unique(x))
    create_segmentation_gif(x, y, x_channel=0, dim=0)
    exit()
    images = add_segmentation_to_image(x, y)
    for image in images:
        print(np.unique(image.astype(np.uint8)))
        plt.imshow(image)
        plt.show()


