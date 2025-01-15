from tqdm import tqdm
import pickle
import os 

gleason_0 = []
gleason_ 3 = []
gleason_4_5 = []


for path in tqdm(paths):
    with open(path, 'rb') as f:
        x = pickle.load(f)
        gleason = x['gleason'][0]
        
        if gleason == '0':
            gleason_0.append(path)
            
        elif gleason == '3':
            gleason_3.append(path)
            
        else:
            gleason_4_5.append(path)
        

with open('/mnt/ip105/dpietrzak/panda/gleason_0_clear_paths.pkl', 'wb') as f:
    pickle.dump(gleason_0, f)
    
with open('/mnt/ip105/dpietrzak/panda/gleason_3_clear_paths.pkl', 'wb') as f:
    pickle.dump(gleason_3, f)

with open('/mnt/ip105/dpietrzak/panda/gleason_4_5_clear_paths.pkl', 'wb') as f:
    pickle.dump(gleason_4_5, f)
