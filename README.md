# Capsule Defect Detection using Autoencoder

This project develops a deep learning model to detect defective pharmaceutical capsules using an **Autoencoder-based anomaly detection approach**.

The model learns the normal visual patterns of capsules and identifies abnormal capsules based on **reconstruction error**.

---

## Features

- Image preprocessing using OpenCV
- Capsule region extraction with contour detection
- Image resizing and normalization
- Autoencoder neural network for anomaly detection
- Model training with cross-validation
- Reconstruction error analysis
- Visualization of anomalies using error maps

---

## Dataset

The dataset consists of capsule images used for quality inspection.

Two main categories:

- Normal Capsules (used for training)
- Defective Capsules (used for anomaly detection testing)

---

## Data Preprocessing

The following preprocessing steps were applied:

- Capsule region extraction using contour detection
- Image resizing to **256 × 256**
- RGB color conversion
- Pixel normalization
- Image flattening for neural network input

---

## Deep Learning Workflow

1. Load capsule image dataset
2. Extract capsule region using OpenCV
3. Resize images to a consistent dimension
4. Normalize pixel values
5. Train Autoencoder on normal capsule images
6. Reconstruct images using the trained model
7. Calculate reconstruction error
8. Detect anomalies based on threshold

---

## Model Architecture

The model is based on a **Fully Connected Autoencoder**.

Encoder:
- Dense layer (32 units)
- Dense layer (16 units)
- Dense layer (8 units)

Decoder:
- Dense layer (8 units)
- Dense layer (16 units)
- Dense layer (32 units)
- Output reconstruction layer

Loss Function: Mean Squared Error (MSE)  
Optimizer: Adam

---

## Training Strategy

To improve model generalization:

- **5-Fold Cross Validation**
- **Dropout Regularization**
- **Early Stopping**

Training parameters:

- Epochs: 50
- Batch Size: 64
- Learning Rate: 0.001

---

## Anomaly Detection

The anomaly detection is based on **reconstruction error**.

Steps:

1. Reconstruct the input image using the trained autoencoder
2. Compute the reconstruction error
3. Compare the error with a defined threshold
4. Images exceeding the threshold are classified as **defective capsules**

Threshold formula:

```
threshold = mean reconstruction error + 2 × standard deviation
```

---

## Visualization

The project includes several visualization techniques:

- Original capsule image
- Reconstructed capsule image
- Encoded representation
- Pixel-level anomaly heatmap
- Highlighted defect regions

These visualizations help interpret how the model detects anomalies.

---

## Technologies

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

---

## Notebook

The full implementation is available in:

`Capsule_Defect_Detection_Autoencoder.ipynb`

---

## Run on Kaggle

You can run the project directly on Google Colab:

[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue)](https://www.kaggle.com/code/achittaphonsumkham/projectcomvis1)

---

## Applications

This anomaly detection approach can be applied in:

- Pharmaceutical quality inspection
- Manufacturing defect detection
- Industrial visual inspection
- Medical anomaly detection
