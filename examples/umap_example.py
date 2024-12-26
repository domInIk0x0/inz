import numpy as np
import matplotlib.pyplot as plt
import umap

np.random.seed(42) 
num_points = 100  # Liczba punktów na klaster

# Klaster A - wokół (0, 0, 0)
cluster_A = np.random.normal(loc=[0, 0, 0], scale=1, size=(num_points, 3))

# Klaster B - wokół (10, 10, 10)
cluster_B = np.random.normal(loc=[10, 10, 10], scale=1, size=(num_points, 3))

# Klaster C - wokół (0, 10, 0)
cluster_C = np.random.normal(loc=[0, 10, 0], scale=1, size=(num_points, 3))

# Połączenie klastrów
all_points = np.vstack([cluster_A, cluster_B, cluster_C])
labels = np.array([0] * num_points + [1] * num_points + [2] * num_points)

# Wizualizacja w 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(cluster_A[:, 0], cluster_A[:, 1], cluster_A[:, 2], label='Cluster A', alpha=0.6)
ax.scatter(cluster_B[:, 0], cluster_B[:, 1], cluster_B[:, 2], label='Cluster B', alpha=0.6)
ax.scatter(cluster_C[:, 0], cluster_C[:, 1], cluster_C[:, 2], label='Cluster C', alpha=0.6)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title('3D Clusters')
plt.show()

# UMAP na danych 3D
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
umap_embedding = reducer.fit_transform(all_points)

# Wizualizacja UMAP
plt.figure(figsize=(8, 6))
plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.colorbar(label='Cluster Label')
plt.title('UMAP Projection')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.show()
