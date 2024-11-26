import os
import cv2
from skimage.metrics import structural_similarity as ssim
import pickle
from tqdm import tqdm

def calculate_ssim(imageA, imageB):
    grayA = cv2.cvtColor(imageA, cv2.COLOR_RGB2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_RGB2GRAY)
    score, _ = ssim(grayA, grayB, full=True)
    return score


def compare_tiles_lr(tiles, lr_tile, output_path):
    with open(os.path.join(tiles, lr_tile), 'rb') as f:
        data = pickle.load(f)
        img_A = data['patch']

    paths = os.listdir(tiles)

    for path in tqdm(paths):
        full_path = os.path.join(tiles, path)
        with open(full_path, 'rb') as f:
            data = pickle.load(f)
            img_B = data['patch']

        score = calculate_ssim(imageA=img_A, imageB=img_B)
        info['path'].append(full_path)
        info['ssim'].append(score)

    with open(output_path, 'wb') as f:
        pickle.dump(info, f)


info = {'path': [],
       'ssim': []}

tiles = '/mnt/ip105/dpietrzak/panda/normalized_tiles/'
lr_tile = 'normalized_tile_96389.pkl'
output_path = '/mnt/ip105/dpietrzak/panda/lr_tiles_compare.pkl'
compare_tiles_lr(tiles, lr_tile, output_path)
