# Movie Ratings Analysis and Recommendation System
This repository includes:

## movie_titles.csv
The dataset containing movie titles, years, and IDs.

## main.py
The main script for analyzing movie ratings, preprocessing data, clustering, and building a recommendation system.

## Usage
### Data Loading and Preprocessing:

Load the movie ratings dataset and perform initial preprocessing.
Handle missing values and reset indices.
### Data Visualization:

Visualize the distribution of ratings, number of ratings per movie, number of ratings per user, and a heatmap of ratings.
### Data Cleaning:

Remove rows with missing ratings and assign movie IDs.
Trim the dataset based on benchmarks for minimum reviews per movie and user.
### Feature Extraction and Clustering:

Extract features from the movie titles dataset.
Standardize the features and handle missing values.
Apply KMeans clustering to the standardized features and visualize the results.
### Recommendation System:

Use the Surprise library to build and evaluate a recommendation system using SVD.
Predict ratings for a specific user and recommend top movies.
