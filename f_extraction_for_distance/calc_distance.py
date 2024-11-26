import pickle
import pandas as pd
import torch
from tqdm import tqdm
from umap import UMAP


with open('/mnt/ip105/dpietrzak/panda/extracted_features.pkl', 'rb') as f:
    df = pickle.load(f)

features = torch.tensor(df.drop(columns=['label', 'image_id', 'path']).values, device='cuda')  # Dane na GPU
labels = torch.tensor(df['label'].values, device='cuda')
image_ids = df['image_id'].values
image_paths = df['path'].values


umap = UMAP(n_components=2, random_state=42)
reduced_features = umap.fit_transform(features.cpu().numpy())
reduced_features = torch.tensor(reduced_features, device='cuda')


reduced_df = pd.DataFrame(reduced_features.cpu().numpy(), columns=['dim1', 'dim2'])
reduced_df['label'] = labels.cpu().numpy()
reduced_df['image_id'] = image_ids
reduced_df['path'] = image_paths


class_0 = reduced_df[reduced_df['label'] == 0]
class_3 = reduced_df[reduced_df['label'] == 3]


class_0_features = torch.tensor(class_0[['dim1', 'dim2']].values, device='cuda')
class_3_features = torch.tensor(class_3[['dim1', 'dim2']].values, device='cuda')

results = []


for idx, (id_0, path_0, features_0) in tqdm(
        enumerate(zip(class_0['image_id'], class_0['path'], class_0_features)), total=len(class_0)):


    distances = torch.norm(class_3_features - features_0, dim=1)


    min_idx = torch.argmin(distances).item()
    max_idx = torch.argmax(distances).item()

    results.append({
        'image_id_0': id_0,
        'image_path_0': path_0,
        'min_distance': distances[min_idx].item(),
        'min_image_id_3': class_3.iloc[min_idx]['image_id'],
        'min_image_path_3': class_3.iloc[min_idx]['path'],
        'max_distance': distances[max_idx].item(),
        'max_image_id_3': class_3.iloc[max_idx]['image_id'],
        'max_image_path_3': class_3.iloc[max_idx]['path']
    })


results_df = pd.DataFrame(results)
results_df.to_pickle('/mnt/ip105/dpietrzak/panda/distance_results.pkl')
