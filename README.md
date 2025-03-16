# 3D Vision Transformer for Brain Tumor Segmentation
The repository contains a Pytorch implementation of a 3D Vision Transformer, by adapting a 2D ViT I implemented in an earlier project (link). The architecture is based on UNETR, and uses skip connections to process multi-scale information

The 3D ViT was then used to effectively perform Tumor Segmentation on MRIs from the BraTS2020 dataset, utilizing. full-resolution 3D MRIs.

<img src="images/1.gif" />
<img src="images/2.gif"/> 
<img src="images/3.gif"/>

*A GIF of an example of a predicted segmentation from the test-set*

## The Architecture
A traditional Vision Transformer works by cutting the original image in patches (eg. 8x8), generating a learned embedding for each patch, adding a positional embedding to each patch embedding, and then using several transformer layers as described in *"Attention Is All You Need"

A 3D Vision Transformer simply uses 3D patches (eg 8x8x8), instead of 2D patches. In the UnetR architecture, skip connections, with attached feature extraction blocks, are connected between parts of the encoder and decoder in a "U" like fashion. This allows the model to process multi-scale information, and combats the loss of finer detail as the network deepens.

![alt text](image.png)
*Visualization of the UnetR architecture, sourced from the UnetR paper by Hatamizadeh et al*

## Tumor Segmentation: BraTS2020
![alt text](images/val.png)

*Slices of an example prediction segmentation from the test-set, compared with the ground truth*

## Limitations
Because of the immense memory usage and long training time, network size had to be downgraded from the original UNetR paper, even with a batch size of 0. I had to use less layers, a bigger patch size and a lower embedding size.

Furthermore, Because of the training time. I did not fine-tune any hyperparameters apart from playing witht the learning rate. Transformations are also basic and unoptimized.

Nevertheless, I am very happy with the performance, but there are definitely improvements left on the table.
