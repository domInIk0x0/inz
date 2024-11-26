import pickle
import pandas as pd
import torch
from tqdm import tqdm
from umap import UMAP

with open('/mnt/ip105/dpietrzak/panda/extracted_features.pkl', 'rb') as f:
    df = pickle.load(f)

features = torch.tensor(df.drop(columns=['label', 'image_id', 'path']).values, device='cuda')
labels = torch.tensor(df['label'].values, device='cuda')
image_ids = df['image_id'].values
image_paths = df['path'].values

umap_configs = [
    {'n_neighbors': 5, 'min_dist': 0.1},
    {'n_neighbors': 15, 'min_dist': 0.1},
    {'n_neighbors': 15, 'min_dist': 0.5},
    {'n_neighbors': 30, 'min_dist': 0.1},
    {'n_neighbors': 30, 'min_dist': 0.5},
]

umap_results = {}

for i, config in enumerate(tqdm(umap_configs, desc="UMAP Configurations")):
    umap = UMAP(n_components=2, random_state=42, **config)
    reduced_features = umap.fit_transform(features.cpu().numpy())

    reduced_df = pd.DataFrame(reduced_features, columns=['dim1', 'dim2'])
    reduced_df['label'] = labels.cpu().numpy()
    reduced_df['image_id'] = image_ids
    reduced_df['path'] = image_paths

    umap_results[f'config_{i+1}'] = reduced_df

output_path = '/mnt/ip105/dpietrzak/panda/umap_results.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(umap_results, f)

print("Redukcja wymiarowości zakończona i zapisana w jednej strukturze.")
