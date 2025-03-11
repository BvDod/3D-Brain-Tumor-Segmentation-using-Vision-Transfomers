from CustomDataset.brats_dataset import BratsDataset
from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np

def add_segmentation_to_image(x, y, x_channel=1, dim=0):
    x, y = x.numpy(), y.numpy()
    x = x[:,:,:,x_channel].squeeze()
    print(x.shape)
    x = np.take(x, indices = (x.shape[dim]//2), axis=dim)
    y = np.take(y, indices = (y.shape[dim]//2), axis=dim)
    
    print(x.shape, y.shape)
    io.imshow(color.label2rgb(y.astype(int),image=x/1000,colors=[(255,0,0),(0,0,255),(0,255,0)],alpha=0.01, bg_label=0, bg_color=None))
    plt.show()




if __name__ == "__main__":
    dataset = BratsDataset()
    print(dataset.dim_mapping)
    x, y = dataset[90]
    add_segmentation_to_image(x, y)


