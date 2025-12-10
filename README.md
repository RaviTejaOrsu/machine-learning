Self-Supervised Contrastive Learning: A Comparative Study of SimCLR, MoCo, and BYOL

This repository contains the code, figures, and tutorial associated with the report “Self-Supervised Contrastive Learning: A Comparative Study of SimCLR, MoCo, and BYOL Through Temperature-Driven Representation Analysis.”
The project investigates how contrastive objectives, temperature scaling, and architectural components affect the quality and geometry of learned representations using synthetic data.

Repository Structure
/
├── notebooks/
│   └── machine learning code.ipynb
├── figures/
│   ├── pca_tau_0.05.png
│   ├── pca_tau_0.1.png
│   ├── pca_tau_0.5.png
│   ├── pca_tau_1.0.png
│   ├── cm_tau_0.05.png
│   ├── cm_tau_0.1.png
│   ├── cm_tau_0.5.png
│   ├── cm_tau_1.0.png
│   └── byol_confusion_matrix.png
├── tutorial.pdf
├── LICENSE
└── README.md

Project Overview

This project explores three key self-supervised learning frameworks:

SimCLR (contrastive, large-batch training)

MoCo-style queue mechanism (momentum encoder and stored negatives)

BYOL (negative-free, predictor–target approach)

We evaluate how the temperature parameter (τ) in the InfoNCE loss affects embedding separability, gradient sharpness, and the risk of representation collapse. A synthetic 2D dataset is used to provide clear, interpretable visualisations of embedding geometry.

The final analysis includes:

PCA visualisations of embeddings

Confusion matrices from linear evaluation

Comparison between contrastive models and BYOL

Discussion of mutual information and theoretical alignment

How to Run the Notebook
1. Install Requirements
pip install torch torchvision matplotlib seaborn scikit-learn numpy

2. Open the Notebook

From the repository root:

jupyter notebook notebooks/machine learning code.ipynb

3. Run All Cells

The notebook will:

Generate synthetic data

Apply contrastive augmentations

Train models for τ = 0.05, 0.1, 0.5, 1.0

Visualise embeddings with PCA

Compute confusion matrices

Train and evaluate a simplified BYOL model

Save all generated figures to /figures

Dataset Description

The dataset consists of 2,000 samples generated using make_blobs with four Gaussian clusters in two dimensions. Noise-based augmentations simulate the view transformations required for contrastive learning. Labels are used only for linear evaluation and are not provided to the contrastive learner.
