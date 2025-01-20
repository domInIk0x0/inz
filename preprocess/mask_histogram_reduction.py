"""
SKRYPT REDUKUJĄCY ILOŚĆ ZDJĘĆ NA PODSTAWIE 
WARTOŚCI MASEK SEGMENTACYJNYCH NA PODSTAWIE USTALONYCH PROGÓW
"""


import pickle
import os
from tqdm import tqdm

def new_clear_paths_ext(paths, output):
    new_clear_paths = []
    for path in tqdm(paths):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            gleason = data['gleason']
            mask_sum = data['mask_sum']
            provider = data['provider']

            if provider == 'radboud':
                if gleason == '0+0':
                    if mask_sum <= 110000:
                        new_clear_paths.append(path)
                elif gleason == '3+3':
                    if 110000 < mask_sum <= 160000:
                        new_clear_paths.append(path)
                elif gleason == '4+4':
                    if 160000 < mask_sum <= 250000:
                        new_clear_paths.append(path)
                elif gleason == '5+5':
                    if mask_sum > 250000:
                        new_clear_paths.append(path)

            elif provider == 'karolinska':
                if gleason == '3+3':
                    if mask_sum > 110000:
                        new_clear_paths.append(path)
                elif gleason == '4+4':
                    if mask_sum > 100000:
                        new_clear_paths.append(path)
                elif gleason == '5+5':
                    if mask_sum > 100000:
                        new_clear_paths.append(path)

            else:
                new_clear_paths.append(path)

    with open(output, 'wb') as f:
        pickle.dump(new_clear_paths, f)

# Ścieżki do plików wejściowych i wyjściowych
path = '/mnt/ip105/dpietrzak/panda/clear_paths.pkl'
output_path = '/mnt/ip105/dpietrzak/panda/new_clear_paths_poprawka.pkl'

with open(path, 'rb') as f:
    paths = pickle.load(f)

new_clear_paths_ext(paths, output_path)
