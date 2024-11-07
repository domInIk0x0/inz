import os
import pickle
import numpy as np
import tifffile
import pandas as pd
from tqdm import tqdm
import cv2

# Funkcja filtrująca obrazy na podstawie dostępnych masek i wybranych wartości Gleasona (nie każdy obraz posiada maskę ale jest ich bardzo mało)
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


# Funkcja główna
  def check_zero_pixels(path_train, path_train_mask, labels_path, patch_size, gleason_val):
  
  '''
  Przyjmowane parametry:
  labels_path: sciezka do pliku csv w ktorym sa etykiety itp
  mask_path: sciezka gdzie znajduja sie maski do zdjec
  gleason: tablica w której są etykiety gleason i tylko zdjęcia z takim gleason zostaną przetworzone
  info: wyswietla informacje o ilosci id 
  Zwraca:
  Zwracana jest tablica z id zdjęć które posiadają swoje maski i mają podany gleason
  '''
  
  ids_list = clear_image_ids(labels_path=labels_path, mask_path=path_train_mask, gleason=gleason_val, info=True)
  df = pd.read_csv(labels_path)

  df_lookup = df.set_index('image_id')[['data_provider', 'gleason_score']].to_dict(orient='index')
  tile_index = 1

  for image_id in tqdm(ids_list):
      image_mask = tifffile.imread(os.path.join(path_train_mask, f'{image_id}_mask.tiff'))
      image = tifffile.imread(os.path.join(path_train, f'{image_id}.tiff'))

      x_len, y_len, _ = image_mask.shape
      patches = []

        # Zbieranie kafelków bez zerowej wartości w masce
      for x in range(0, x_len, patch_size):
        for y in range(0, y_len, patch_size):
          
          x_end = min(x + patch_size, x_len)
          y_end = min(y + patch_size, y_len)

          mask_patch = image_mask[x:x_end, y:y_end]
          patch = image[x:x_end, y:y_end]

          if mask_patch.shape[:2] != (patch_size, patch_size):
              pad_vertical = patch_size - mask_patch.shape[0]
              pad_horizontal = patch_size - mask_patch.shape[1]
              mask_patch = np.pad(mask_patch, ((0, pad_vertical), (0, pad_horizontal), (0, 0)), mode='constant', constant_values=0)
              patch = np.pad(patch, ((0, pad_vertical), (0, pad_horizontal), (0, 0)), mode='constant', constant_values=255)

          mask_values = mask_patch[:, :, 0].flatten()
          count_zero = np.sum(mask_values == 0)
          sum_mask_values = np.sum(mask_values)

          if count_zero == 0:
              patches.append((patch, sum_mask_values))


          patches = sorted(patches, key=lambda x: x[1], reverse=True)[:25]

        # Przetwarzanie wybranych kafelków
            for patch, sum_mask_values in patches:
              
                data = df_lookup[image_id]
                provider = data['data_provider']
                gleason_score = data['gleason_score']
    
                if gleason_score == 'negative':
                    gleason_score = '0+0'
    
                # Zapis kafelka i jego informacji
                save_path = f'/mnt/ip105/dpietrzak/panda/tiles/tile_{tile_index}.pkl'
                with open(save_path, 'wb') as f:
                    pickle.dump({
                        'patch': patch,
                        'gleason': gleason_score,
                        'image_id': image_id,
                        'provider': provider,
                        'mask_sum': sum_mask_values,
                        'tile_index': tile_index,
                        'cord_x': (x, x_end),
                        'cord_y': (y, y_end),
                    }, f)
                tile_index += 1



train_path = '/mnt/ip105/dpietrzak/train_images/'
mask_path = '/mnt/ip105/dpietrzak/train_label_masks/'
labels_path = '/mnt/ip105/dpietrzak/train.csv'

check_zero_pixels(path_train=train_path,
                               path_train_mask=mask_path,
                               labels_path=labels_path,
                               patch_size=256,
                               gleason_val=['negative', '0+0', '3+3', '4+4', '5+5'])
