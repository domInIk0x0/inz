# ROZKLAD TEST , ILE ZDJEC z danego 3+4 itd zostalo wyciete kafli
import os
import numpy as np
from tqdm import tqdm
import pickle

all_tiles = []
p = '/mnt/ip105/dpietrzak/panda/test_tiles/'
for image in os.listdir(p):
    i = len(os.listdir(os.path.join(p, image)))
    print(i)
    all_tiles.append(i)

all_tiles = np.array(all_tiles)

with open('/mnt/ip105/dpietrzak/panda/test_roz.pkl', 'wb') as f:
    pickle.dump(all_tiles, f)
