import os
import cv2
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(imageA, imageB):
    grayA = cv2.cvtColor(imageA, cv2.COLOR_RGB2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_RGB2GRAY)
    score, _ = ssim(grayA, grayB, full=True)
    return score

def compare_tiles_lr(tiles_dir, lr_tile_path, output_path):
    with open(lr_tile_path, 'rb') as f:
        lr_data = pickle.load(f)
        lr_image = lr_data['patch']

    info = {'path': [], 'ssim': []}

    tile_paths = os.listdir(tiles_dir)
    for tile_path in tqdm(tile_paths, desc="Processing tiles"):
        full_path = os.path.join(tiles_dir, tile_path)

        with open(full_path, 'rb') as f:
            tile_data = pickle.load(f)
            tile_image = tile_data['patch']

        score = calculate_ssim(imageA=lr_image, imageB=tile_image)
        info['path'].append(full_path)
        info['ssim'].append(score)

    with open(output_path, 'wb') as f:
        pickle.dump(info, f)

def filter_tiles(ssim_data_path, clear_paths_path, output_clear_paths_path, ssim_threshold=0.25):
    with open(ssim_data_path, 'rb') as f:
        ssim_data = pickle.load(f)

    with open(clear_paths_path, 'rb') as f:
        clear_paths = pickle.load(f)

    df = pd.DataFrame(ssim_data)
    filtered_paths = df[df['path'].isin(clear_paths) & (df['ssim'] < ssim_threshold)]['path'].values

    with open(output_clear_paths_path, 'wb') as f:
        pickle.dump(filtered_paths, f)

    return len(filtered_paths)

tiles_dir = '/mnt/ip105/dpietrzak/panda/normalized_tiles/'
lr_tile_path = '/mnt/ip105/dpietrzak/panda/normalized_tile_96389.pkl'
ssim_data_path = '/mnt/ip105/dpietrzak/panda/lr_tiles_compare.pkl'
clear_paths_path = '/mnt/ip105/dpietrzak/panda/clear_paths.pkl'
output_clear_paths_path = '/mnt/ip105/dpietrzak/panda/clear_paths.pkl'

compare_tiles_lr(tiles_dir, lr_tile_path, ssim_data_path)

num_filtered = filter_tiles(ssim_data_path, clear_paths_path, output_clear_paths_path)
print(f"Number of filtered paths: {num_filtered}")
