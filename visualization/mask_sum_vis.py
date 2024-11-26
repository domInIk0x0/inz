import matplotlib.pyplot as plt
import pickle
import pandas as pd

with open('/mnt/data/dpietrzak/panda/clear_paths.pkl', 'rb') as f:
    paths = pickle.load(f)
    
d = {'path': [],
    'provider': [],
    'gleason': [],
    'mask_sum': []}

for i in paths:
    with open(i.replace('/ip105/','/data/'), 'rb') as f:
        p = pickle.load(f)
        d['path'].append(i)
        d['provider'].append(p['provider'])
        d['gleason'].append(p['gleason'])
        d['mask_sum'].append(p['mask_sum'])

        
t = pd.DataFrame(d)
plt.hist((t[(t['gleason']== '5+5') & (t['provider'] == 'radboud')].mask_sum))

t3 = t[(t['gleason']== '3+3') & (t['provider'] == 'radboud')]
t4 = t[(t['gleason']== '4+4') & (t['provider'] == 'radboud')]
t5 = t[(t['gleason']== '5+5') & (t['provider'] == 'radboud')]

plt.hist(t3.mask_sum, alpha=0.5, label='gleason 3', color='blue')
plt.hist(t4.mask_sum, alpha=0.5, label='gleason 4', color='yellow')
plt.hist(t5.mask_sum, alpha=0.5, label='gleason 5', color='red')
