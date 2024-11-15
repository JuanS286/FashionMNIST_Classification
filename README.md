# Fashion MNIST Classification with Convolutional Neural Networks (CNNs)

This repository contains a project on classifying images from the Fashion MNIST dataset using Convolutional Neural Networks (CNNs) implemented in Python with TensorFlow and Keras. The notebook demonstrates a step-by-step approach to building and evaluating a deep learning model.

## Table of Contents
- [Dataset](#dataset)
- [Project Overview](#project-overview)
- [Model Architecture](#model-architecture)
- [Key Features](#key-features)
- [Results](#results)
- [How to Use](#how-to-use)
- [Dependencies](#dependencies)
- [Acknowledgments](#acknowledgments)

## Dataset
The Fashion MNIST dataset consists of 70,000 grayscale images of size 28x28 pixels, categorized into 10 classes of clothing items:
- Training Set: 60,000 images
- Test Set: 10,000 images

The dataset is preloaded in Keras and does not require additional downloads.

## Project Overview
The goal of this project is to classify Fashion MNIST images into one of the 10 clothing categories using a CNN model. Key steps in the notebook include:
1. Data loading and preprocessing.
2. Exploratory data analysis (EDA) and visualization.
3. Building the CNN model with multiple layers.
4. Training and validating the model.
5. Evaluating the model on the test dataset.
6. Visualizing predictions.

## Model Architecture
The implemented CNN architecture includes:
- Convolutional layers for feature extraction.
- Max pooling layers for dimensionality reduction.
- Dropout layers for regularization.
- Dense layers for classification.
- Softmax activation for final output.

## Key Features
- **Early Stopping**: Integrated to prevent overfitting.
- **Dropout**: Used to enhance model generalization.
- **Visualization**: Includes plots for accuracy, loss, and sample predictions.
- **Comparison of Models**: Evaluated multiple models to choose the best one.

## Results
- Best Model Accuracy: ~91% on the test dataset.
- Loss and accuracy plots highlight the training process.
- Correctly classified images and misclassifications are displayed with predictions.


