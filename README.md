# Multi-Attention Enhanced Encoder-Decoder Network with Hybrid Transformer Bottleneck for Echocardiography Image Segmentation

Official implementation of the Scientific Reports submission:

**“Multi-Attention Enhanced Encoder-Decoder Network with Hybrid Transformer Bottleneck for Echocardiography Image Segmentation”**

## Overview

This repository contains the implementation of a deep learning framework for automatic segmentation of 2D echocardiography images using a hybrid CNN–Transformer architecture.

The proposed model combines:

* Vision Transformer (ViT)
* Multi Receptive Field Block (MRFB)
* Convolutional Block Attention Module (CBAM)
* Squeeze-and-Excitation (SE) blocks
* Attention Gates
* Atrous Spatial Pyramid Pooling (ASPP)
* Deep Supervision

The framework is designed to improve both local feature extraction and global contextual modeling for robust cardiac structure segmentation in echocardiography images.

---

## Dataset

This project uses the publicly available CAMUS dataset:

---

## Segmentation Targets

The model performs segmentation for:

* Left Ventricle Endocardium (LVendo)
* Left Ventricle Epicardium (LVepi)
* Left Atrium (LA)

for both:

* Apical Two-Chamber (2CH)
* Apical Four-Chamber (4CH)

views.

---

## Proposed Architecture

The architecture is based on an encoder-decoder framework with:

* CNN-based encoder
* Hybrid bottleneck:

  * Vision Transformer (ViT)
  * Multi Receptive Field Block (MRFB)
* Attention-enhanced skip connections
* Deep supervision with ASPP modules

The model aims to capture:

* Fine-grained local spatial details
* Long-range global dependencies
* Multi-scale contextual information

---

## Requirements

Example environment:

```bash
Python 3.10
TensorFlow 2.15
Keras
NumPy
OpenCV
scikit-learn
matplotlib
```

---

## Training

Example training command:

```bash
python train.py
```

Training settings used in the paper:

* Optimizer: Adam
* Initial learning rate: 1e-4
* Batch size: 8
* Epochs: 100
* 5-fold cross-validation

---

## Evaluation Metrics

The model is evaluated using:

* Dice Similarity Coefficient (DSC)
* Jaccard Index (IoU)
* Hausdorff Distance (HD)
* Precision
* Recall
* Specificity
* F2-score

---

## Results

The proposed method achieved strong segmentation performance on the CAMUS dataset.

Example Dice scores:

| Structure | 2CH Dice (%) | 4CH Dice (%) |
| --------- | ------------ | ------------ |
| LVendo    | 91.11        | 92.25        |
| LVepi     | 87.60        | 87.29        |
| LA        | 87.85        | 91.71        |

---




## DOI

Zenodo DOI:

```text
10.5281/zenodo.20440298
```

---

## License

This project is provided for academic and research purposes.

---

## Contact

Saeed Chamani
Department of Biomedical Engineering
Iran University of Science and Technology

GitHub:
https://github.com/saeedchamani


