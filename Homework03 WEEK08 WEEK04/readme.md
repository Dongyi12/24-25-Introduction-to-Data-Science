# Flower Classification and Clustering
# This repository includes:

## main.py
The main script for training a ResNet model on a flower dataset, extracting features, and performing clustering analysis.

## Usage
Setup and Data Preparation:

Load the flower dataset and apply transformations for training and validation.
Ensure the dataset path exists and is correctly specified.
Data Loading:

Load training and validation datasets using torchvision.datasets.ImageFolder.
Create data loaders for efficient batch processing.
Model Setup:

Load a pre-trained ResNet-34 model and modify the final fully connected layer to match the number of flower classes.
Move the model to the appropriate device (GPU or CPU).
## Training:

Train the model using the training dataset.
Validate the model using the validation dataset.
Save the best model based on validation accuracy.
## Feature Extraction:

 Extract features from the trained model for both training and validation datasets.

## Clustering Analysis:

Apply PCA to reduce feature dimensions.
Perform KMeans clustering on the reduced features.
Visualize the clustering results using scatter plots.

