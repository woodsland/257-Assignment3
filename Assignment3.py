# COMP257 - Unsupervised & Reinforcement Learning (Section 002)
# Assignment 3 - Hierarchical Clustering
# Name: Wai Lim Leung
# ID  : 301276989
# Date: 14-Oct-2023

from sklearn.datasets import fetch_olivetti_faces
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Part 1 & 2- Retrieve & Load Olivetti Faces
olivetti_faces = fetch_olivetti_faces(shuffle=True, random_state=42)
of_data = olivetti_faces.data
of_target = olivetti_faces.target
of_images = olivetti_faces.images
of_labels = olivetti_faces.target

print()
print("Part 1 & 2 - Olivetti Faces")
print("Data Shape  :", of_data.shape)
print("Target Shape:", of_target.shape)

fig = plt.figure(figsize=(6, 2))
for i in range(3):
    face_image = of_data[i].reshape(64, 64)
    position = fig.add_subplot(1, 3, i + 1)
    position.imshow(face_image, cmap='gray')
    position.set_title(f"Person {of_target[i]}")
    position.axis('off')
plt.tight_layout()
plt.show()

# Part 3a - Training, Validation & Test
X_temp, X_test, y_temp, y_test = train_test_split(of_images, of_labels, test_size=0.2, stratify=of_labels, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)
print()
print("Part 3")
print("Training  :", [np.sum(y_train == i) for i in range(40)])
print("Validation:", [np.sum(y_val == i)   for i in range(40)])
print("Test      :", [np.sum(y_test == i)  for i in range(40)])

# Part 3b - Cluster
n_clusters = 240
cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
X_train_flat = X_train.reshape((X_train.shape[0], -1))
clusters = cluster.fit_predict(X_train_flat)
print("Cluster   :")
print(clusters)

# Part 4 - kFold, Train Classifier & Predict
X_train_kfold = X_train.reshape((X_train.shape[0], -1))
X_val_kfold = X_val.reshape((X_val.shape[0], -1))
classf = RandomForestClassifier(n_estimators=100, random_state=42)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Training classifier using k-fold cross-validation
cross_val_scores = cross_val_score(classf, X_train_kfold, y_train, cv=kfold)
print()
print("Part 4 - Scores")
print("Cross Validation   :", cross_val_scores)
print("Average            : ", np.mean(cross_val_scores))

# Train and validate
classf.fit(X_train_flat, y_train)
val_predictions = classf.predict(X_val_kfold)
val_accuracy = accuracy_score(y_val, val_predictions)
print("Validation Accuracy: ", val_accuracy)

# Part 5 - Centroid Based Clustering
def perform_clustering(data, affinity_matrix=None):
    if affinity_matrix is None:
        return AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='average').fit_predict(data)
    else:
        return AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='average').fit_predict(affinity_matrix)

# 5a - Euclidean Distance
euclidean_affinity = euclidean_distances(X_train_flat)
euclidean_clusters = perform_clustering(X_train_flat, euclidean_affinity)

# 5b - Minkowski Distance
minkowski_affinity_matrix = squareform(pdist(X_train_flat, metric='minkowski', p=3))
minkowski_clusters = perform_clustering(X_train_flat, minkowski_affinity_matrix)

# 5c - Cosine Similarity
cosine_affinity = cosine_distances(X_train_flat)
cosine_clusters = perform_clustering(X_train_flat, cosine_affinity)

# 5d - Result
print()
print("Part 5 - Centroid Based Clustering")
print("5a. Euclidean Distance:")
print(euclidean_clusters)

print()
print("5b. Minkowski Distance:")
print(minkowski_clusters)

print()
print("5c. Cosine Similarity :")
print(cosine_clusters)

# Part 6 - Silhouette Score
def optimal_clusters(data, affinity_matrix=None):
    range_clusters = list(range(2, 41))
    best_score = -1
    best_n_clusters = 2

    for n_clusters in range_clusters:
        if affinity_matrix is None:
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='average')
            preds = clusterer.fit_predict(data)
            score = silhouette_score(data, preds)
        else:
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='average')
            preds = clusterer.fit_predict(affinity_matrix)
            score = silhouette_score(affinity_matrix, preds, metric='precomputed')

        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters

    return best_n_clusters, best_score

# 6a - Euclidean Distance
best_n_clusters_euclidean, best_score_euclidean = optimal_clusters(X_train_flat, euclidean_affinity)
print()
print("Part 6 - Silhouette Score")
print("Euclidean Distance - Numer of Clusters:", best_n_clusters_euclidean)
print("                   - Silhouette Score :", best_score_euclidean)

# 6b - Minkowski Distance
best_n_clusters_minkowski, best_score_minkowski = optimal_clusters(X_train_flat, minkowski_affinity_matrix)
print("Minkowski Distance - Numer of Clusters:", best_n_clusters_minkowski)
print("                   - Silhouette Score :", best_score_minkowski)

# 6c - Cosine Similarity
best_n_clusters_cosine, best_score_cosine = optimal_clusters(X_train_flat, cosine_affinity)
print("Cosine Similarity  - Numer of Clusters:", best_n_clusters_cosine)
print("                   - Silhouette Score :", best_score_cosine)