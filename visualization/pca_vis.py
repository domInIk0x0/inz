"""
SKRYPT WIZUALIZUJĄCY ZREDUKOWANĄ PRZESTRZEŃ CECH
UZYSKANĄ Z SIECI BEZ OSTATNIEJ WARSTWY. PCA
"""


import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

with open('/mnt/data/dpietrzak/panda/extracted_features.pkl', 'rb') as f:
    df = pickle.load(f)

features = df.drop(columns=['label', 'image_id', 'path']).values  
labels = df['label'].values 

class_palette = {
    0: 'green',       # Klasa 0
    3: 'yellow',      # Klasa 3
    4: 'darkorange',  # Klasa 4
    5: 'red'          # Klasa 5
}


def plot_2d(features_2d, labels, title):
    """Tworzy wykres 2D zredukowanych wymiarów."""
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=features_2d[:, 0],
        y=features_2d[:, 1],
        hue=labels,
        palette=class_palette,
        alpha=0.7,
        s=50 
    )
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(title="Class")
    plt.grid(True)
    plt.show()


def plot_3d(features_3d, labels, title):
    """Tworzy wykres 3D zredukowanych wymiarów."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    unique_labels = np.unique(labels)
    for label in unique_labels:
        idx = labels == label
        ax.scatter(
            features_3d[idx, 0],
            features_3d[idx, 1],
            features_3d[idx, 2],
            label=f"Class {label}",
            color=class_palette[label],
            alpha=0.7,
            s=50  
        )
    ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_zlabel("Dimension 3")
    ax.legend(title="Class")
    plt.show()

pca = PCA(n_components=3)
pca_2d = pca.fit_transform(features)[:, :2] 
pca_3d = pca.fit_transform(features) 

explained_variance = pca.explained_variance_ratio_[:2]
print(f"PCA - Wyjaśniona wariancja (2D): {explained_variance}")

plot_2d(pca_2d, labels, "PCA - Redukcja do 2 wymiarów")
plot_3d(pca_3d, labels, "PCA - Redukcja do 3 wymiarów")
