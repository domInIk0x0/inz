train_path = '/mnt/data/dpietrzak/train_images/'
mask_path = '/mnt/data/dpietrzak/train_label_masks/'
labels_path = '/mnt/data/dpietrzak/train.csv'

train_df = pd.read_csv(labels_path)
gleason_all = train_df.groupby('gleason_score').gleason_score.count()
gleason_clear = train_df[train_df['gleason_score'].isin(['0+0','3+3', '4+4', '5+5'])].groupby('gleason_score').gleason_score.count()
gleason_clear_sum = gleason_clear.sum()
provider_count = train_df.groupby('data_provider').data_provider.count()

print(f'Ilość zdjęć dla danego gleason {gleason_all} \n')
print(f'Ilość zdjęć tylko dla wybranych gleason {gleason_clear} \n')
print(f'Suma zdjęć dla wybranych etykiet gleason {gleason_clear_sum}')
print(f'Ilość zdjęc dostarczonych przez konktretnego dostawcę {provider_count}')
print(f' Rozkład zdjęć {train_df.groupby(['data_provider', 'gleason_score']).count().image_id)}')
