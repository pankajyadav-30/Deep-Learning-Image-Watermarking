# Deep Learning Based Image Watermarking using Dual-Path Autoencoders

Deep Learning-based image watermarking system using GrabCut segmentation, dual-path autoencoders for robust object/background watermark embedding and image recombination for secure copyright protection.

## Overview

This project implements a robust image watermarking framework using Deep Learning and Computer Vision techniques.

The system first separates an image into Object and Background regions using GrabCut segmentation. Independent autoencoders are then used to embed watermark information into both regions. Finally, the regions are recombined to produce a watermarked image while preserving visual quality.

## Features

- Image partitioning using GrabCut
- Object and Background segmentation
- Dual-path watermark embedding
- Autoencoder-based watermark generation
- Reconstruction of final watermarked image
- Training on PASCAL VOC 2012 dataset
- GPU accelerated training using CUDA

## Dataset

PASCAL VOC 2012 Dataset

Dataset statistics:

Total Images Parsed: 17,125
Training Images Used: 1,600
Validation Images Used: 400
Batch Size: 8

Dataset downloaded using KaggleHub:

```python
import kagglehub

path = kagglehub.dataset_download(
    "huanghanchina/pascal-voc-2012"
)

print(path)
