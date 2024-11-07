import os
import pickle
import numpy as np
import cv2
from tqdm import tqdm
import torchstain
from skimage.metrics import structural_similarity as ssim
import shutil


def calculate_ssim(imageA, imageB):
    '''
    OPIS: 
    Funkcja oblicza ssim pomiędzy dwoma zdjęciami w skali szarości 
    w tym przypadku (znormalizowany, i nie znormalizowany)
    
    PARAMETRY:
    ImageA: obraz1
    ImageB: obraz2
    
    ZWRACA:
    SSIM SCORE 
    '''
    grayA = cv2.cvtColor(imageA, cv2.COLOR_RGB2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_RGB2GRAY)
    score, _ = ssim(grayA, grayB, full=True)
    return score


def check_single_tile(tile, threshold=0.8):
    '''
    OPIS: 
    Funkcja oblicza procent tla w kafelku
    
    PARAMETRY:
    tile: Kafelek 256x256x3
    threshold: Powyzej jakiej wartosci jest uznawane cos za tlo
    
    ZWRACA:
    SSIM SCORE 
    '''
    
    tile_gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
    normalized_array = tile_gray / 255.0
    white_pixels = np.sum(normalized_array > threshold)
    prc = white_pixels / normalized_array.size
    return prc


def check_h_e(tile):
    '''
    OPIS: 
    Funkcja oblicza procent hematoksyliny i eozyny
    a bardziej sprawdza ile pikseli jest w zakresie tych kolorow.
    Zakresy to dolny zakres koloru i gorny 
    Funkcje o bardzo niskich wartosciach np (0, 0) to kafelki z samym tlem
    albo inne dziwne rzeczy
    
    PARAMETRY:
    tile: Kafelek 256x256x3
    
    ZWRACA:
    Krotke z h&e (h, e)  
    '''
    
    hsv_tile = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
    hematoxylin_lower = np.array([100, 50, 50])
    hematoxylin_upper = np.array([130, 255, 255])
    
    eosin_lower = np.array([140, 50, 50])
    eosin_upper = np.array([180, 255, 255])
    
    hematoxylin_mask = cv2.inRange(hsv_tile, hematoxylin_lower, hematoxylin_upper)
    eosin_mask = cv2.inRange(hsv_tile, eosin_lower, eosin_upper)
    
    hematoxylin_percentage = (cv2.countNonZero(hematoxylin_mask) / hematoxylin_mask.size) * 100
    eosin_percentage = (cv2.countNonZero(eosin_mask) / eosin_mask.size) * 100
    
    return hematoxylin_percentage, eosin_percentage


def process_tiles(input_path, output_path, reference_path, info_output_path):
    '''
    OPIS: 

    
    PARAMETRY:
    input_path:  sciezka do folderu gdzia znajduja sie pociete kafelki przez funkcje crop_all_images
    output_path: sciezka gdzie maja zostac zapisane znormalizowane kafelki 
    reference_path: sciezka do kafelka ktory jest uzywany jako kafelek wzorcowy do normalizacji MACENKO
    info_output_path: sciezko gdzie jest zapisywana tablica z takimi informacjami jak procent_tla, suma_maski itd.
    
    ZWRACA:
    Funkcja zapisuje znormalizowane kafelki do podanego folderu i dodatkowo zapisuje tablice z roznymi informacjami
    '''
    
    with open(reference_path, 'rb') as file:
        data = pickle.load(file)
        reference_tile = data['patch']
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend='numpy')
    normalizer.fit(reference_tile)

    os.makedirs(output_path, exist_ok=True)

    tiles_info = {
        'image_id': [], 'data_provider': [], 'mask_sum': [], 'gleason_score': [],
        'background_percent': [], 'H&E': [], 'SSIM': [], 'is_normalized': []
    }

    tile_files = os.listdir(input_path)
    for idx, tile_file in enumerate(tqdm(tile_files, desc="Processing tiles")):
        with open(os.path.join(input_path, tile_file), 'rb') as file:
            data = pickle.load(file)
            patch = data['patch']
            image_id = data['image_id']
            gleason = data['gleason']
            provider = data['provider']
            mask_sum = data['mask_sum']
            tile_index = data['tile_index']
            cord_x = data['cord_x']
            cord_y = data['cord_y']

            try:
                normalized_tile, _, _ = normalizer.normalize(I=patch, stains=True)
                is_normalized = True
            except Exception as e:
                print(f"Normalization failed for tile {tile_file}: {e}")
                normalized_tile = patch
                is_normalized = False

            output_file = os.path.join(output_path, f"normalized_tile_{idx}.pkl")
            with open(output_file, 'wb') as out_file:
                pickle.dump({
                    'patch': normalized_tile,
                    'gleason': gleason,
                    'image_id': image_id,
                    'provider': provider,
                    'mask_sum': mask_sum,
                    'tile_index': tile_index,
                    'cord_x': cord_x,
                    'cord_y': cord_y
                }, out_file)

            prc = check_single_tile(normalized_tile)
            ssim_score = calculate_ssim(patch, normalized_tile)
            pen_trace = detect_pen_percentage(patch)
            h, e = check_h_e(patch)

            tiles_info['image_id'].append(image_id)
            tiles_info['data_provider'].append(provider)
            tiles_info['mask_sum'].append(mask_sum)
            tiles_info['gleason_score'].append(gleason)
            tiles_info['background_percent'].append(prc)
            tiles_info['H&E'].append((h, e))
            tiles_info['SSIM'].append(ssim_score)
            tiles_info['is_normalized'].append(is_normalized)

    with open(info_output_path, 'wb') as info_file:
        pickle.dump(tiles_info, info_file)

    print("Processing complete.")

    
def select_representative_tiles(tiles_info_path, clear_ids):
    '''
    OPIS:
    
    PARAMETRY:
    tiles_info_path: sciezka gdzie znajduje sie tablica z informacjami na temat procentu tła itd.
    clear_ids: sciezka gdzie zostanie zapisana tablica ze sciezkami do treningu
    
    ZWRACA:
    Sciezki do zdjec które zostana uzyte do treningu
    '''
    with open(info_path, 'rb') as f:
        data = pickle.load(f)
        
    info = pd.DataFrame(data)
    
    # Dodanie kolumny zeby wiedziec o numerze kafelka
    info['path_to_tile'] = info.index.to_series().apply(lambda x: f'/mnt/data/dpietrzak/panda/normalized_tiles/normalized_tile_{x}.pkl')
    info = info[(info['is_normalized'] == True) & (info['background_percent'] <= 0.2)]
    info['H&E_sum'] = info['H&E'].apply(lambda x: sum(x))
    info = info[(info['H&E_sum'] >= 60)] 
    info = info[(info['SSIM'] >= 0.9)] 
    
    clear_path = np.array(info.path_to_tile)

    # Zapisanie sciezek do zdjec które zostaną użyte do treningu
    with open(clear_ids, 'wb') as f:
        pickle.dump(clear_path, f)

        
def copy_files_to_folder(file_paths, destination_folder):
    '''
    OPIS: Funkcja która przenosi zdjęcia które zostały 
    wybrane do treningu z folderu ze wszystkimi znormalizowanymi kafelkami do osobnego folderu.
    
    PARAMETRY:
    file_paths: sciezka do tablicy ze sciezkami z wybranymi zdjeciami do treningu
    destination_folder: sciezka pod jaka maja zostac zapisane zdjecia treningowe
    
    ZWRACA:
    
    '''
    os.makedirs(destination_folder, exist_ok=True)
    with open(file_paths, 'rb') as f:
        file_paths = pickle.load(f)

    for file_path in file_paths:
        if os.path.exists(file_path):
            shutil.copy(file_path, destination_folder)
            print(f"Copied {file_path} to {destination_folder}")
        else:
            print(f"File {file_path} does not exist and was skipped.")
            

input_path = '/mnt/ip105/dpietrzak/panda/tiles'
output_path = '/mnt/ip105/dpietrzak/panda/normalized_tiles'
reference_path = '/mnt/ip105/dpietrzak/panda/tiles/tile_159065.pkl'
info_output_path = '/mnt/ip105/dpietrzak/panda/tiles_info.pkl'
file_paths = '/mnt/ip105/dpietrzak/panda/clear_paths.pkl'
destination_folder = '/mnt/ip105/dpietrzak/panda/clear_tiles'

process_tiles(input_path, output_path, reference_path, info_output_path)
select_representative_tiles(info_output_path, file_paths)
copy_files_to_folder(file_paths, destination_folder)

