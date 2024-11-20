import pickle
import cv2
from skimage.metrics import structural_similarity as ssim
import tifffile
from tqdm import tqdm


def calculate_ssim(imageA, imageB):
    grayA = cv2.cvtColor(imageA, cv2.COLOR_RGB2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_RGB2GRAY)
    score, _ = ssim(grayA, grayB, full=True)
    return score


with open('/mnt/ip105/dpietrzak/panda/tiles_4_5_ssim_list.pkl', 'rb') as f:
    info = pickle.load(f)

paths_4_4 = [
    tile for gleason, tile in zip(info['gleason_score'], info['path_to_tile']) if gleason == '4+4'
]
paths_5_5 = [
    tile for gleason, tile in zip(info['gleason_score'], info['path_to_tile']) if gleason == '5+5'
]

print(f"Found {len(paths_4_4)} tiles for Gleason score 4+4")
print(f"Found {len(paths_5_5)} tiles for Gleason score 5+5")

similarities = []
i = 1

for path_4 in tqdm(paths_4_4, desc="Processing 4+4 tiles"):
    with open(path_4, 'rb') as f:
        data = pickle.load(f)
    image_4 = data['patch']

    for path_5 in paths_5_5:
        with open(path_5, 'rb') as f:
            data = pickle.load(f)
        image_5 = data['patch']


        score = calculate_ssim(image_4, image_5)
        similarities.append({
            'path_4+4': path_4,
            'path_5+5': path_5,
            'ssim_score': score
        })

    i+= 1
    if i == 50:
        output_path = '/mnt/ip105/dpietrzak/panda/ssim_results_g4_g5.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(similarities, f)
            print("Processing complete.")
            break
