import nibabel as nib
import nilearn as nil
import matplotlib.pyplot as plt
from nilearn import plotting
import numpy as np

image_names = ["t1", "t1ce", "t2", "flair", "seg"]
images = []

for name in image_names:
    images.append(nib.load(f'dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_003/BraTS20_Training_003_{name}.nii'))

brain_vol = images[0]

# Numpy array of shape (240, 240, 155)
data = brain_vol.get_fdata()

image_plot = data[:,:,70]
for image in images[1:]:
    print(image.get_fdata()[:,:,70])
    image_plot = np.concatenate((image_plot,image.get_fdata()[:,:,70]),axis=1)

# Show one slice (horizontal)
plt.imshow(image_plot, cmap="bone")
plt.show()

# Show one slice (horizontal)
plt.imshow(images[-1].get_fdata()[:,:,70], cmap="bone")
plt.show()



# grid of slices
fig_rows = 4
fig_cols = 4
n_subplots = fig_rows * fig_cols
n_slice = data.shape[-1]
step_size = n_slice // n_subplots
plot_range = n_subplots * step_size
start_stop = int((n_slice - plot_range) / 2)
fig, axs = plt.subplots(fig_rows, fig_cols, figsize=[10, 10])

for idx, img in enumerate(range(start_stop, plot_range, step_size)):
    axs.flat[idx].imshow(data[:, :, img], cmap='gray')
    axs.flat[idx].axis('off')       
plt.tight_layout()
plt.show()


# plot using neuroimaging specific library
plotting.plot_img(brain_vol)
plt.show()