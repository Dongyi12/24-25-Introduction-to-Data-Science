# Diabetes Data Analysis and Model Training
This repository includes:

## diabetes.csv
The dataset used for analysis and model training.

## main.py
The main script for data analysis, preprocessing, clustering, and model training.

## Usage
Load and View Data:

Load the dataset and display basic information, including the first few rows, data types, and summary statistics.
## Data Visualization:

Visualize the distribution of the target variable (Outcome).
Plot histograms for each feature to understand their distributions.
Create a heatmap to visualize the correlation between features.
## Data Preprocessing:

Separate the features and target variable.
Split the data into training and testing sets.
Standardize the features using StandardScaler.
Cluster Analysis:

Apply KMeans clustering to the standardized training data.
Visualize the clustering results using a scatter plot.
Decision Tree Model:

Train a decision tree classifier on the standardized training data.
Predict the target variable for the test data.
Evaluate the model using accuracy score, classification report, and confusion matrix.
Visualize the confusion matrix using a heatmap.
Decision Tree Visualization:

Visualize the trained decision tree model using the dtreeviz library.
