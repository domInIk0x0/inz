import pickle
import pandas as pd
import numpy as np
from umap import UMAP

# Ramka z wektorem z resneta18 (512x1) plus etykiety, image_id, i sciezka do zdjecia
with open('/mnt/data/dpietrzak/panda/extracted_features.pkl', 'rb') as f:
    df = pickle.load(f)

features = df.drop(columns=['label', 'image_id', 'image_path']).values
labels = df['label'].values
image_ids = df['image_id'].values
image_paths = df['image_path'].values

umap = UMAP(n_components=2, random_state=42)
reduced_features = umap.fit_transform(features)

reduced_df = pd.DataFrame(reduced_features, columns=['dim1', 'dim2'])
reduced_df['label'] = labels
reduced_df['image_id'] = image_ids
reduced_df['image_path'] = image_paths

class_0 = reduced_df[reduced_df['label'] == 0]
class_3 = reduced_df[reduced_df['label'] == 3]

# Obliczanie odległości pomiędzy punktami z klasy 0 i klasy 3
results = []
for _, row_0 in class_0.iterrows():
    id_0, path_0, features_0 = row_0['image_id'], row_0['image_path'], np.array([row_0['dim1'], row_0['dim2']])
    distances = []
    for _, row_3 in class_3.iterrows():
        id_3, path_3, features_3 = row_3['image_id'], row_3['image_path'], np.array([row_3['dim1'], row_3['dim2']])
        dist = np.linalg.norm(features_0 - features_3)
        distances.append((dist, id_3, path_3))

    # Znajdowanie najmniejszej i największej odległości
    min_dist, min_id_3, min_path_3 = min(distances, key=lambda x: x[0])
    max_dist, max_id_3, max_path_3 = max(distances, key=lambda x: x[0])

    results.append({
        'image_id_0': id_0,
        'image_path_0': path_0,
        'min_distance': min_dist,
        'min_image_id_3': min_id_3,
        'min_image_path_3': min_path_3,
        'max_distance': max_dist,
        'max_image_id_3': max_id_3,
        'max_image_path_3': max_path_3
    })


results_df = pd.DataFrame(results)
results_df.to_csv('/mnt/data/dpietrzak/panda/distance_results.csv', index=False)
