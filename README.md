# Mango Shelf-Life Classification Using CNN and Vision Transformer Models with Explainable AI

Automated five-stage mango ripeness classification using ImageNet-pretrained CNN and Vision Transformer models, with Grad-CAM, Eigen-CAM, and LIME used to explain every model's predictions.

## Overview
Mango ripeness is usually judged by eye, which is inconsistent and leads to post-harvest loss. This project classifies mango images into five shelf-life stages: **Unripe, Semi-ripe, Fully Ripe, Overripe, and Perished**. Five deep learning models are trained under one identical setup and compared.

## Dataset
- **Mango Shelf-Life Dataset** (Mashkoor and Hakim, 2024), originally published on Mendeley Data and accessed through a Kaggle mirror
- 4,428 images of the White Chaunsa Late mango variety, taken under controlled studio lighting
- Imbalanced classes (Unripe 1,350, Semi-ripe 1,157, Overripe 896, Fully Ripe 841, Perished 184)
- Stratified 70/15/15 train/validation/test split
- The dataset is not included in this repository. Please download it from the original source.

## Methods
- **Models:** ResNet50, EfficientNet-B0, DenseNet121 (CNN) and DeiT-Small, Swin-Tiny (Transformer), all with ImageNet-pretrained transfer learning
- **Class imbalance:** handled with a class-weighted loss function
- **Ensembles:** soft voting and accuracy-weighted voting of all five models
- **Explainable AI:** Grad-CAM, Eigen-CAM, and LIME applied to every model
- **Framework and hardware:** PyTorch, Kaggle Notebooks with 2× NVIDIA Tesla T4 GPUs

## Results

| Model | Test Accuracy | Test Loss |
|---|---|---|
| ResNet50 | 97.59% | 0.0799 |
| EfficientNet-B0 | 97.14% | 0.0759 |
| DenseNet121 | 97.44% | 0.0610 |
| DeiT-Small | 97.59% | 0.0678 |
| **Swin-Tiny** | **98.20%** | 0.0590 |
| Ensemble (Soft Voting) | 97.89% | 0.0583 |
| Ensemble (Accuracy-Weighted) | 97.89% | 0.0583 |

Results are single-run numbers on the held-out 15% test split.

## How to Run
1. Download the dataset from the source above.
2. Open `NOTEBOOK_NAME.ipynb` in Kaggle or Google Colab (GPU recommended).
3. Update the dataset path in the notebook and run all cells.

