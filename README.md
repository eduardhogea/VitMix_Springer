# VitMix

Code behind the paper: **"ViTmiX: Vision Transformer Explainability Augmented by Mixed Visualization Methods"**.

The paper is only possible thanks to the advancements made in [Transformer-Explainability](https://github.com/hila-chefer/Transformer-Explainability).

> **Note:** This repo requires cloning [Transformer-Explainability](https://github.com/hila-chefer/Transformer-Explainability) and using a compatible ViT model provided therein.

---

## Overview

**ViTmiX** is a framework that improves Vision Transformer interpretability by fusing multiple explainability methods, producing more class-specific and faithful heatmaps. It combines GradCAM, Saliency, Attention Rollout, and LRP using geometric mean and element-wise multiplication.

![image](https://github.com/user-attachments/assets/707fc71c-1d45-42b1-b7db-fa41db398e1f)


## Features

- ⚙️ **Attribution Mixing**: Fusion of multiple attribution methods using multiplication/geometric mean.
- 📊 **Quantitative Evaluation**: IoU, F1 Score, Pixel Accuracy, and Deletion AUC metrics on VOC and ImageNet.
- 📐 **Pigeonhole Theoretical Justification**: Formal proof of why fusion improves explainability.
- 🧠 **Human-Aligned Validation**: Student-annotated masks used for comparison.

---

## Usage
Run the main script by specifying the target dataset:


```bash
python vitmix.py <imagenet|voc>
```

## Results

ViTmiX is evaluated on both ImageNet (student-annotated) and Pascal VOC datasets. Below is a summary of segmentation performance across multiple metrics. The best-performing method across both datasets is the two-way combination of **LRP + Rollout**, using geometric mean fusion.

### Table: Segmentation Metrics for All Methods

| **Method**                     | **Dataset**   | **IoU (%)** | **F1 Score (%)** | **Pixel Accuracy (%)** | **Deletion AUC** |
|-------------------------------|---------------|-------------|------------------|------------------------|------------------|
| **1–Way Methods**             |               |             |                  |                        |                  |
| GradCAM                       | ImageNet      | 11.62       | 17.64            | 15.92                  | 0.59             |
|                               | Pascal VOC    | 7.72        | 13.69            | 16.14                  | 0.65             |
| LRP                           | ImageNet      | 37.91       | 51.96            | 50.51                  | 0.46             |
|                               | Pascal VOC    | 36.31       | 52.04            | 63.05                  | 0.53             |
| Rollout                       | ImageNet      | 38.53       | 53.18            | 67.31                  | 0.44             |
|                               | Pascal VOC    | 28.92       | 44.37            | 56.72                  | 0.54             |
| Saliency                      | ImageNet      | 6.31        | 10.67            | 9.87                   | 0.62             |
|                               | Pascal VOC    | 14.84       | 24.41            | 16.43                  | 0.65             |
| **2–Way Methods**             |               |             |                  |                        |                  |
| LRP + GradCAM                 | ImageNet      | 23.49       | 34.40            | 34.31                  | 0.49             |
|                               | Pascal VOC    | 10.73       | 18.69            | 23.72                  | 0.59             |
| LRP + Rollout                 | ImageNet      | **49.19**   | **63.37**        | **68.51**              | **0.43**         |
|                               | Pascal VOC    | **40.61**   | **56.58**        | **70.13**              | **0.55**         |
| LRP + Saliency                | ImageNet      | 33.20       | 47.31            | 50.78                  | 0.47             |
|                               | Pascal VOC    | 33.43       | 48.62            | 56.77                  | 0.61             |
| Rollout + GradCAM             | ImageNet      | 20.76       | 30.44            | 33.92                  | 0.52             |
|                               | Pascal VOC    | 10.64       | 18.33            | 27.03                  | 0.61             |
| Saliency + GradCAM            | ImageNet      | 14.49       | 21.94            | 22.76                  | 0.58             |
|                               | Pascal VOC    | 8.88        | 15.12            | 16.39                  | 0.61             |
| Saliency + Rollout            | ImageNet      | 18.37       | 28.86            | 38.69                  | 0.54             |
|                               | Pascal VOC    | 30.44       | 44.58            | 52.74                  | 0.62             |
| **3–Way Methods**             |               |             |                  |                        |                  |
| LRP + Rollout + GradCAM       | ImageNet      | 24.50       | 35.53            | 33.54                  | 0.48             |
|                               | Pascal VOC    | 12.58       | 21.42            | 24.02                  | 0.61             |
| LRP + Saliency + GradCAM      | ImageNet      | 19.11       | 28.91            | 27.87                  | 0.54             |
|                               | Pascal VOC    | 10.52       | 18.21            | 22.20                  | 0.60             |
| LRP + Saliency + Rollout      | ImageNet      | 36.15       | 50.29            | 51.16                  | 0.45             |
|                               | Pascal VOC    | 37.44       | 53.81            | 61.49                  | 0.59             |
| Saliency + Rollout + GradCAM  | ImageNet      | 15.83       | 23.97            | 24.96                  | 0.56             |
|                               | Pascal VOC    | 9.57        | 16.32            | 17.19                  | 0.63             |

---


![image](https://github.com/user-attachments/assets/3fd05f9d-efbf-4822-8b3d-c3ced7a5fefd)

![image](https://github.com/user-attachments/assets/f74995a7-a7d2-4caf-8554-ec050b4cad51)


## Citation

If you use this project in your research, please cite:

```bibtex
@inproceedings{,
  title     = {ViTmiX: Vision Transformer Explainability Augmented by Mixed Visualization Methods},
  author    = {},
  booktitle = {XXXXXXX.XXXXXXX},
  year      = {2025},
  doi       = {XXXXXXX.XXXXXXX}
}

