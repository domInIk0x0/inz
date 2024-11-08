import os
from tqdm import tqdm
import numpy as np
import tifffile


def clear_image_ids(labels_path, mask_path, gleason, info=False):
    '''
    Przyjmowane parametry:
    labels_path: sciezka do pliku csv w ktorym sa etykiety itp
    mask_path: sciezka gdzie znajduja sie maski do zdjec
    gleason: tablica w której są etykiety gleason i tylko zdjęcia z takim gleason zostaną przetworzone
    info: wyswietla informacje o ilosci id 
    Zwraca:
    Zwracana jest tablica z id zdjęć które posiadają swoje maski i mają podany gleason
    '''
    
    train_df = pd.read_csv(labels_path)
    gleason_clear = train_df[train_df['gleason_score'].isin(gleason)]
    clear_ids = np.array(gleason_clear['image_id'])
    mask_ids = {path[:-10] for path in os.listdir(mask_path)}
    clear_ids_filtered = [img_id for img_id in clear_ids if img_id in mask_ids]

    if info:
        print(f'Liczba zdjęć z danym gleason score {len(clear_ids)}')
        print(f'Liczba masek segmentacyjnych {len(mask_ids)}')
        print(f'Liczba zdjęć z danym gleason score, które posiadają swoje maski segmentacyjne: {len(clear_ids_filtered)}')

    return clear_ids_filtered


def V(channel):
    return np.mean(channel) / np.std(channel)


def tiles_coefficient_of_variation(ids, train_path, output_path):

    image_info = {
        'image_id': [],
        'r_v': [],
        'g_v': [],
        'b_v': []
    }
    
    for image_id in tqdm(ids):
        image = tifffile.imread(os.path.join(train_path, f'{image_id}.tiff'))
   
        r_v, g_v, b_v = V(image[:, :, 0]), V(image[:, :, 1]), V(image[:, :, 2])
        
        image_info['image_id'].append(image_id)
        image_info['r_v'].append(r_v)
        image_info['g_v'].append(g_v)
        image_info['b_v'].append(b_v)
        
    with open(output_path, 'wb') as f:
        pickle.dump(image_info, f)
        
        

train_path = '/mnt/ip105/dpietrzak/train_images/'
output_path = '/mnt/ip150/dpietrzak/panda/coef_of_var_info.pkl'
mask_path = '/mnt/ip105/dpietrzak/train_label_masks/'
labels_path = '/mnt/ip105/dpietrzak/train.csv'

gleason_val=['negative', '0+0', '3+3', '4+4', '5+5']
ids_list = clear_image_ids(labels_path=labels_path, mask_path=mask_path, gleason=gleason_val, info=True)

tiles_coefficient_of_variation(ids_list=ids_list, train_path=train_path, output_path=output_path)
