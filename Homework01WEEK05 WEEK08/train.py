import matplotlib

matplotlib.use('TkAgg')

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Read in csv
df = pd.read_csv("data/mental_health_analysis.csv")

# See columns
df.isna().any()

corr = df.corr(numeric_only=True)
corr.style.background_gradient(cmap='coolwarm')
x = df
x = x.dropna()
x = x.drop("Academic_Performance", axis=1)
x = x.drop("Gender", axis=1)

x['Support_System'] = x['Support_System'].map({'Low': 0., 'Moderate': 1., 'High': 2.})
x_scaled = x.drop("User_ID", axis=1)
x_scaled = x_scaled.drop("Age", axis=1)
# x_scaled = x_scaled.drop("Academic_Performance", axis=1)
print(x_scaled)

# x_scaled['Support_System'] = x_scaled['Support_System'].map({'Low': 0., 'Moderate': 1., 'High': 2.})
x_scaled = StandardScaler().fit_transform(x_scaled)

# Get reduced dimensions
pca = PCA(n_components=2)
x_2d = pca.fit_transform(x_scaled)

# Plot
plt.figure(figsize=(8, 8))
plt.plot(x_2d[:, 0], x_2d[:, 1], "bx")
plt.title("PCA-Reduced Data")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()


def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)


def plot_centroids(centroids, weights=None, circle_color='r', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=35, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=2, linewidths=12,
                color=cross_color, zorder=11, alpha=1)


def plot_clusters(clusterer, X):
    labels = clusterer.predict(X)
    pca = PCA(n_components=2)
    x_2d = pca.fit_transform(X)
    plt.scatter(x_2d[:, 0], x_2d[:, 1], c=labels, alpha=0.3)
    plot_centroids(clusterer.cluster_centers_)


scores = []
for i in range(1, 20):
    # Fit for k
    means = KMeans(n_clusters=i)
    means.fit(x_scaled)
    # Get inertia
    scores.append(means.inertia_)

print(scores)
plt.plot(scores, "-rx")
plt.xticks(np.arange(0, 21, 1.0))

k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(x_scaled)
plt.figure(figsize=(8, 8))
# Plot clusters onto PCA reduced plot
plot_clusters(kmeans, x_scaled)

# How many dimensions to reduce to (before clustering)?
num_dimensions = 2

# Reduce dimensions
pca = PCA(n_components=num_dimensions)
x_less_dimensions = pca.fit_transform(x_scaled)

# Fit cluster
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(x_less_dimensions)

# Plot results on 2D plot
plt.figure(figsize=(8, 8))
plot_clusters(kmeans, x_less_dimensions)

# Add cluster labels as extra column in dataframe
labels = kmeans.predict(x_less_dimensions)
x["cluster"] = labels

features = ['Social_Media_Hours', 'Exercise_Hours',
            'Sleep_Hours', 'Screen_Time_Hours', 'Survey_Stress_Score',
            'Wearable_Stress_Score', 'Support_System']
width = 1 / (len(features))

fig, ax = plt.subplots(figsize=(12, 8))

for i in range(k):
    ax.bar([i], [1], width=width * 4, color="azure" if i % 2 == 0 else "whitesmoke")

cmap = plt.cm.get_cmap('cool')
# Iterate through features
for index, f in enumerate(features):
    # Get mean for each feature for each cluster
    data = [np.mean(x[x["cluster"] == i][f]) for i in range(k)]

    x_vals = np.arange(len(data)) + (width * index) - 0.5 + width / 2

    # Plot this feature for each cluster
    ax.bar(x_vals, data, width=width, label=f, color=cmap(index / len(features)))

ax.legend()
ax.set_xlabel("cluster number")
categories = ["0", "1", "2", "3", "4"]

features = ['Social_Media_Hours', 'Exercise_Hours', 'Sleep_Hours', 'Screen_Time_Hours',
            'Survey_Stress_Score', 'Wearable_Stress_Score', 'Support_System']

try:
    plt.style.use("premium.mplstyle")
except:
    pass
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.facecolor"] = "#f1f5f9"

colors = ["#8e93af", "#d7a6b3", "#e8cda5", "#9dc1c5", "#123467"]

fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
x_offset = 0.0
print(x.values)
for i, (category, subset) in enumerate(zip(categories, x.values)):
    print(i, category, subset)
    parts = ax.violinplot(subset, positions=[i + x_offset],
                          showmeans=False,
                          showmedians=False,
                          showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.4)
    jitter = np.random.normal(0, 0.04, size=subset.shape)
    ax.scatter(np.full(subset.shape, i + x_offset) + jitter, subset,
               color=colors[i], edgecolor="#888888", s=5)
    ax.boxplot(subset, positions=[i], widths=0.2, patch_artist=True,
               boxprops=dict(facecolor=colors[i], color="#f1f5f9", alpha=0.75),
               capprops=dict(color=colors[i]),
               whiskerprops=dict(color=colors[i]),
               flierprops=dict(markeredgecolor=colors[i]),
               medianprops=dict(color="#f1f5f9"))

ax.set_xticks(range(len(categories)))
ax.set_xticklabels(categories)
ax.set_title("Composite Box Diagram", x=0.015, y=0.95, ha="left", va="top")
plt.tight_layout()
plt.show()
