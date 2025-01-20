"""
SKRYPT WIZUALIZUJĄCY JAKIE OBRAZY Z DANYM ID SĄ NAJCZĘŚCIEJ W PARZE Z DANYMI
OBRAZAMI.
"""
import pandas as pd
import matplotlib.pyplot as plt

results_path = '/mnt/data/dpietrzak/panda/distance_results.pkl'
results_df = pd.read_pickle(results_path)

max_distance_counts = results_df['max_image_path_3'].value_counts()
min_distance_counts = results_df['min_image_path_3'].value_counts()

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
max_distance_counts.plot(kind='bar', color='red')
plt.title('Histogram obrazów w maksymalnym dystansie')
plt.xlabel('ID obrazu (max)')
plt.ylabel('Liczba wystąpień')
plt.xticks(rotation=90)

plt.subplot(1, 2, 2)
min_distance_counts.plot(kind='bar', color='green')
plt.title('Histogram obrazów w minimalnym dystansie')
plt.xlabel('ID obrazu (max)')
plt.ylabel('Liczba wystąpień')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

most_frequent_max_image = max_distance_counts.idxmax()
most_frequent_max_count = max_distance_counts.max()

most_frequent_min_image = min_distance_counts.idxmax()
most_frequent_min_count = min_distance_counts.max()

print("=== Obrazy najczęściej w parze z maksymalnym dystansem ===")
print(f"Obraz ID: {most_frequent_max_image}, liczba wystąpień: {most_frequent_max_count}")

print("\n=== Obrazy najczęściej w parze z minimalnym dystansem ===")
print(f"Obraz ID: {most_frequent_min_image}, liczba wystąpień: {most_frequent_min_count}")
