"""
HISTOGRAMY DLA PROCENTU TŁA, SSIM, H&E
"""


import pickle
import matplotlib.pyplot as plt
import pandas as pd


path_info = '/mnt/data/dpietrzak/panda/tiles_info.pkl'
with open(path_info, 'rb') as f:
    info = pickle.load(f)

# HISTOGRAM: Background Percent dla wszystkich kafelków
plt.figure(figsize=(10, 8))
plt.hist(info['background_percent'], bins='auto', color='blue', alpha=0.7, edgecolor='black')
plt.title("ALL TILES: Background Percent")
plt.xlabel("Background Percent")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Funkcja do tworzenia histogramów dla Gleason Score i data_provider
def plot_histograms(info, column, title_prefix, xlabel):
    fig, axs = plt.subplots(4, 2, figsize=(12, 16))
    gleason_scores = ['0+0', '3+3', '4+4', '5+5']
    providers = ['radboud', 'karolinska']

    for i, score in enumerate(gleason_scores):
        for j, provider in enumerate(providers):
            data = info[(info['gleason_score'] == score) & (info['data_provider'] == provider)][column]
            axs[i, j].hist(data, bins='auto', color='blue', alpha=0.7, edgecolor='black')
            axs[i, j].set_title(f"{title_prefix} {score} {provider.capitalize()}")
            axs[i, j].set_xlabel(xlabel)
            axs[i, j].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

# HISTOGRAMY: Background Percent dla każdej grupy
plot_histograms(info, column='background_percent', title_prefix="GLEASON", xlabel="Background Percent")

# HISTOGRAM: SSIM dla wszystkich kafelków
plt.figure(figsize=(10, 8))
plt.hist(info['SSIM'], bins='auto', color='blue', alpha=0.7, edgecolor='black')
plt.title("ALL TILES: SSIM")
plt.xlabel("SSIM")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# HISTOGRAMY: SSIM dla każdej grupy
plot_histograms(info, column='SSIM', title_prefix="GLEASON", xlabel="SSIM")

# Sumowanie kanałów H&E
def calculate_he_sum(x):
    return sum(x) if isinstance(x, list) else 0

info['H&E_sum'] = info['H&E'].apply(calculate_he_sum)
plt.figure(figsize=(10, 8))
plt.hist(info['H&E_sum'], bins='auto', color='green', alpha=0.7, edgecolor='black')
plt.title("H&E Sum")
plt.xlabel("Sum of H&E Channels")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
counts.plot(kind='bar', color='orange', alpha=0.7, edgecolor='black')
plt.title("Number of Tiles by Gleason Score")
plt.xlabel("Gleason Score")
plt.ylabel("Number of Tiles")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
