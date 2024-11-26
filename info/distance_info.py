import pandas as pd

results_path = '/mnt/data/dpietrzak/panda/distance_results.pkl'
results_df = pd.read_pickle(results_path)

min_distances = results_df.nsmallest(10, 'min_distance')  
max_distances = results_df.nlargest(10, 'max_distance')  

print("=== 10 przypadków z najmniejszymi odległościami ===")
for idx, row in min_distances.iterrows():
    print(f"Image ID 0: {row['image_id_0']}")
    print(f"Image Path 0: {row['image_path_0']}")
    print(f"Min Distance: {row['min_distance']}")
    print(f"Min Image ID 3: {row['min_image_id_3']}")
    print(f"Min Image Path 3: {row['min_image_path_3']}\n")

print("\n=== 10 przypadków z największymi odległościami ===")
for idx, row in max_distances.iterrows():
    print(f"Image ID 0: {row['image_id_0']}")
    print(f"Image Path 0: {row['image_path_0']}")
    print(f"Max Distance: {row['max_distance']}")
    print(f"Max Image ID 3: {row['max_image_id_3']}")
    print(f"Max Image Path 3: {row['max_image_path_3']}\n")
