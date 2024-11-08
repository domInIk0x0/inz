import os
import pickle
import numpy as np
import tifffile
import pandas as pd
from tqdm import tqdm
import cv2

def clear_ids_test(test_path, labels_path, gleason=['4+3', '4+5', '3+4', '5+4', '5+3', '3+5'], info=False):
    train_df = pd.read_csv(labels_path)
    gleason_clear = train_df[train_df['gleason_score'].isin(gleason)]
    clear_ids = np.array(gleason_clear['image_id'])

    if info:
        print(f'Liczba zdjęć z danym gleason score {len(clear_ids)}')

    return clear_ids


def segment_tissue(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary_mask = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(binary_mask)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    tissue_only = cv2.bitwise_and(image, image, mask=mask)
    return tissue_only, mask


def resize_with_padding(image, target_size=(256, 256)):
    h, w = image.shape[:2]
    top, bottom = 0, target_size[0] - h
    left, right = 0, target_size[1] - w

    padded_image = cv2.copyMakeBorder(
        image,
        top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255)
    )

    return padded_image


def prepare_test_set(path_test, labels_path, patch_size, gleason_val):

    clear_ids_test = clear_image_ids(test_path=path_test, labels_path=labels_path, gleason=gleason_val, info=True)
    df = pd.read_csv(labels_path)
    df_lookup = df.set_index('image_id')[['data_provider', 'gleason_score']].to_dict(orient='index')

    for image_id in tqdm(ids_list):
        image = tifffile.imread(os.path.join(path_test, f'{image_id}.tiff'))
        tissue_image, mask = segment_tissue(image)
        x_len, y_len = tissue_image.shape[:2]

        patches = []

        for x in range(0, x_len, patch_size):
            for y in range(0, y_len, patch_size):
                x_end = min(x + patch_size, x_len)
                y_end = min(y + patch_size, y_len)

                mask_patch = mask[x:x_end, y:y_end]
                patch = tissue_image[x:x_end, y:y_end]

                tissue_ratio = np.sum(mask_patch > 0) / (patch_size * patch_size)

                if tissue_ratio >= 0.75:
                    if patch.shape[:2] != (patch_size, patch_size):
                        patch = resize_with_padding(patch, (patch_size, patch_size))

                    patches.append(patch)


        save_folder = os.path.join('/mnt/ip105/dpietrzak/panda/test_tiles', image_id)
        os.makedirs(save_folder, exist_ok=True)

        print(f'Zapisano {len(patches)} kafelków z dana iloscia tkanki.')
        
        for i, patch in enumerate(patches):
            data = df_lookup[image_id]
            provider = data['data_provider']
            gleason_score = data['gleason_score']

            save_path = os.path.join(save_folder, f'tile_{i+1}.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'patch': patch,
                    'gleason': gleason_score,
                    'image_id': image_id,
                }, f)


test_path = '/mnt/ip105/dpietrzak/train_images/'
labels_path = '/mnt/ip105/dpietrzak/train.csv'

prepare_test_set(path_test=test_path,
                 labels_path=labels_path,
                 patch_size=256,
                 gleason_val=['4+3', '4+5', '3+4', '5+4', '5+3', '3+5'])

