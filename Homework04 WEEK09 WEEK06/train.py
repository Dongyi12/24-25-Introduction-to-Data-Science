import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from dtreeviz import model  # Using the new API

# Set out data
data = pd.read_csv('diabetes.csv')

# View basic data information
print(data.head())  # View the first 5 rows
print(data.info())  # View Data Information
print(data.describe())  # View Statistics

# Checking for missing values
print(data.isnull().sum())  # Check for missing values

# Data visualisation
# 1. Distribution of target variables
plt.figure(figsize=(6, 4))
sns.countplot(x='Outcome', data=data)
plt.title('Distribution of Outcome')
plt.show()

# 2. characteristic distribution
data.hist(figsize=(12, 10))
plt.show()

# 3. characteristic correlation
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()

# Data preprocessing
# 1. Separation of characteristics and target variables
X = data.drop('Outcome', axis=1)  # diagnostic property
y = data['Outcome']  # target variable

# 2. Data set segmentation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Standardised features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Converting X_train_scaled and X_test_scaled to DataFrame
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

# Cluster Analysis
# Clustering of features using KMeans
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_train_scaled_df)

# Visualisation of clustering results
plt.figure(figsize=(8, 6))
plt.scatter(X_train_scaled_df.iloc[:, 0], X_train_scaled_df.iloc[:, 1], c=clusters, cmap='viridis', s=50)
plt.title('KMeans Clustering (First 2 Features)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Training decision tree models
dt_model = DecisionTreeClassifier(random_state=42, max_depth=3) 
# Modify the variable name to dt_model
dt_model.fit(X_train_scaled_df, y_train)

# Prediction using converted test data
y_pred = dt_model.predict(X_test_scaled_df)

# assessment model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visual Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Visualising decision trees with dtreeviz
viz_model = model(dt_model, X_train_scaled_df, y_train, target_name='Outcome',
                  feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'])
viz_model.view()