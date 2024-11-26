def apply_mask(mask, data_provider):
    #################################################################################
    ### Funkcja która służy do stworzenia mapy kolorów na podstawie maski oraz    ###
    ### dostawcy danych aby móc wyswietlac maski przy uzyciu plt.show(maska)      ###
    #################################################################################

    # Utworzenie pustej mapy kolorów
    color_mask = np.zeros(shape=(mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # Mapa kolorow na podstawie dostawcy danych
    if data_provider == 'radboud':
        color_map_radboud = {
            0: [0, 0, 0],           # 0: Tło (bez tkanki) lub nieznane
            1: [144, 238, 144],     # 1: stroma (tkanka łączna, tkanka niebędąca nabłonkiem)
            2: [0, 128, 0],         # 2: zdrowy (łagodny) nabłonek
            3: [255, 255, 0],       # 3: nabłonek nowotworowy (Gleason 3)
            4: [255, 165, 0],       # 4: nabłonek nowotworowy (Gleason 4)
            5: [255, 0, 0]          # 5: nabłonek nowotworowy (Gleason 5)
        }
        color_map = color_map_radboud
    else:
        color_map_karolinska = {
            0: [0, 0, 0],           # tło - szary
            1: [0, 255, 0],         # tkanka łagodna - zielony
            2: [255, 0, 0]          # tkanka rakowa - czerwony
        }
        color_map = color_map_karolinska

    # Przypisanie poszczegolnym pikseleom maski dany kolor w zaleznosci od wartosci slownika
    for label, color in color_map.items():
        color_mask[mask[:, :, 0] == label] = color

    return color_mask


    def crop_single_image(image_path, path_train, path_train_mask, labels_path, patch_size, n_tiles):
    #################################################################################
    ### Funkcja która służy do pociecia pojedynczego zdjęcia na n kawałkow o      ###
    ### ustalonych rozmiarach                                                     ###
    #################################################################################

    # Wczytanie pliku csv z danymi, przypisanie etykiet
    train_csv = pd.read_csv(labels_path)
    image_id = image_path[:-10]
    provider = train_csv[train_csv['image_id'] == image_id]['data_provider'].iloc[0]
    gleason_score = train_csv[train_csv['image_id'] == image_id]['gleason_score'].iloc[0]
    isup = train_csv[train_csv['image_id'] == image_id]['isup_grade'].iloc[0]

    # Wczytanie obrazu
    image_mask = tifffile.imread(os.path.join(path_train_mask, image_path))
    image = tifffile.imread(os.path.join(path_train, f'{image_id}.tiff'))

    # Listy do przechowywania fragmentów maski, zdjecia, wartosci sumy maski
    patches = []
    masked_patches = []
    info_scores = []

    x_len, y_len, _ = image.shape
    
    kafelek = 1 
    for x in range(0, x_len, patch_size):
        for y in range(0, y_len, patch_size):
            x_end = min(x + patch_size, x_len)
            y_end = min(y + patch_size, y_len)

            patch = image[x:x_end, y:y_end]
            mask_patch = image_mask[x:x_end, y:y_end]

            # Dodanie pustego tła jeśli obraz nie spełnia zadanych rozmiarów
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                pad_vertical = patch_size - patch.shape[0]
                pad_horizontal = patch_size - patch.shape[1]
                patch = np.pad(
                    patch,
                    ((0, pad_vertical), (0, pad_horizontal), (0, 0)),  
                    mode='constant',
                    constant_values=0
                )
            
            
            if mask_patch.shape[0] < patch_size or mask_patch.shape[1] < patch_size:
                pad_vertical = patch_size - mask_patch.shape[0]
                pad_horizontal = patch_size - mask_patch.shape[1]
                mask_patch = np.pad(
                    mask_patch,
                    ((0, pad_vertical), (0, pad_horizontal), (0, 0)),  
                    mode='constant',
                    constant_values=0
                )

            # Sumowanie wartosci maski 
            info_score = np.sum(mask_patch[:, :, 0].flatten())
            info_scores.append((info_score, patch, mask_patch))

    # Sortowanie wartosci sumy mask 
    info_scores.sort(key=lambda x: x[0], reverse=True)
    selected_patches = info_scores[:n_tiles]

    # Dodanie tylko istotnych kafelkow
    for score, patch, mask_patch in selected_patches:
        patches.append(patch)
        masked_patch = apply_mask(mask_patch, provider)
        masked_patches.append(masked_patch)

    return patches, masked_patches, isup, provider, gleason_score


def plot_patches_no_gap(patches, n_tiles):
    #################################################################################
    ### Funkcja która służy do wyswietlenia pociętych fragmentów obrazów w siatce ###
    #################################################################################
    
    grid_size = int(np.sqrt(n_tiles)) 

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))

    for i, patch in enumerate(patches):
        row = i // grid_size
        col = i % grid_size

        # Obramowanie wokół każdego fragmentu
        axes[row, col].imshow(patch)
        axes[row, col].spines['top'].set_color('black')
        axes[row, col].spines['bottom'].set_color('black')
        axes[row, col].spines['left'].set_color('black')
        axes[row, col].spines['right'].set_color('black')
        axes[row, col].spines['top'].set_linewidth(2)
        axes[row, col].spines['bottom'].set_linewidth(2)
        axes[row, col].spines['left'].set_linewidth(2)
        axes[row, col].spines['right'].set_linewidth(2)

        # Ukryj osie
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        axes[row, col].set_xticklabels([])
        axes[row, col].set_yticklabels([])

    # Usuwanie odstępów między obrazami
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
    plt.show()

image_id = os.listdir(mask_path)[3210]
image_id = image_id[:-10]

plt.figure(figsize=(10, 10))
plt.imshow(tifffile.imread(os.path.join(train_path, f'{image_id[:-10]}.tiff'), key=0))

patches, masked_patches, isup, provider, gleason_score = crop_single_image(image_path=image_id, 
                                                            path_train=train_path, 
                                                            path_train_mask=mask_path, 
                                                            labels_path=labels_path, 
                                                            patch_size = 256, 
                                                            n_tiles=25)
plot_patches_no_gap(patches, n_tiles=25)
plot_patches_no_gap(masked_patches, n_tiles=25)

      
  
