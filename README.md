# Garbage Classification with BERT and ResNet50
Lucas Rodrigues Valença, Matheus Marinho de Morais Leça
ENEL 645 Assignment 2 - Fall 2024

## Overview

This notebook implements a garbage classification task using a hybrid model combining BERT for text processing and ResNet50 for image feature extraction, as the assignment 2 of the ENEL 645 course. The model is trained to classify images into multiple categories (black bin, green bin, blue bin, and other), using a combination of text and image data for more accurate predictions.

### Steps
1. Dataset Loading
2. Model Definition
3. Model Training
4. Prediction and Evaluation

### Requirements
- PyTorch for model training
- Transformers for text encoding using BERT
- Torchvision for ResNet50

## Model Architecture

### ResNet Feature Extraction
We use ResNet50 as the image feature extractor. Pre-trained weights from ResNet50 are used, and the output features are flattened and passed to the classifier.

### BERT Text Encoder
We use BERT to process the textual descriptions associated with the images. The text is tokenized and combined with the image features for the final classification.

### Classifier
We concatenate both image and text features, using a fully connected layer to produce the logits.

## Hyperparameters
- Learning rate: 5e-5
- Batch size: 8
- Number of epochs: 10
- Weight decay: 0.01
- Warm-up steps: 0

## Results and Discussions
At first we planned to use Florence, a multimodal model provided by Microsoft, but we had several compatibility issues when trying to use it on the TALC Cluster. Because of that, we changed our approach by using the architecture mentioned above (ResNet + BERT). Yet, some difficulties were present: we couldn't run the .ipynb file using the GPU on the cluster, only .py files using GPU, even though both were using the same environment and kernel. Because of that, we decided to use Google Colab. We were able to upload only 11gb out of 14gb from the dataset provided on the cluster.

Despite these challenges, we were able to achieve positive results, with an accuracy of 81.8% as demonstrated below:

| Class        | Precision | Recall  | F1 Score | Support |
|--------------|-----------|---------|----------|---------|
| Black        | 0.7638    | 0.6561  | 0.7059   | 695     |
| Blue         | 0.8312    | 0.8258  | 0.8285   | 1085    |
| Green        | 0.8121    | 0.9574  | 0.8788   | 799     |
| TTR          | 0.8477    | 0.8099  | 0.8283   | 852     |
| Accuracy     |           |         | 0.8181   | 0.8181  |
| Macro avg    | 0.8137    | 0.8123  | 0.8104   | 3431    |
| Weighted avg | 0.8172    | 0.8181  | 0.8153   | 3431    |

The classification results shows consistent performance across all classes. It exhibits good accuracy, precision, recall, and F1 scores. As we can see, the “Green” class stands out with high recall (0.96) and F1 score (0.88), indicating the model’s effectiveness in identifying this class. While other classes, such as “Black” and “TTR,” show lower performance, they still demonstrate decent precision and recall values, suggesting potential areas for increasing the performance.
