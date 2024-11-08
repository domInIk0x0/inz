import os
from tqdm import tqdm
import numpy as np


def V(channel):
    mean = np.mean(channel)
    std_dev = np.std(channel)
    cv = std_dev / mean
    
    return cv 


def tiles_coefficient_of_variation(train_path, output_path):
    image_ids = os.listdir(train_path)
    ids_list = list(map(lambda x: x[:-5], image_ids))
    
    image_info = {
        'image_id': [],
        'r_v': [],
        'g_v': [],
        'b_v': []
    }
    
    for image_id in tqdm(ids_list):
        image = tifffile.imread(os.path.join(path_train, f'{image_id}.tiff'))
        
        r = image[:, :, 0]
        g = image[:, :, 1]
        b = image[:, :, 2]
        
        r_v, g_v, b_v = V(r), V(g), V(b)
        
        image_info['image_id'].append(image_id)
        image_info['r_v'].append(r_v)
        image_info['g_v'].append(g_v)
        image_info['b_v'].append(b_v)
        
    with open(output_path, 'wb') as f:
        pickle.dump(f)
        
        
train_path = '/mnt/ip105/dpietrzak/train_images/'
output_path = '/mnt/ip150/dpietrzak/panda/coef_of_var_info.pkl'

tiles_coefficient_of_variation(train_path, output_path)
