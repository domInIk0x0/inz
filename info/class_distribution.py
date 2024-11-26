from collections import Counter
import torch
from torch.utils.data import Dataset, random_split
from torchvision.transforms import transforms
from PIL import Image
import pickle

class GleasonTilesLoader(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        with open(self.directory, 'rb') as f:
            self.files = pickle.load(f)
        self.transform = transform

        self.label_map = {'0': 0, '3': 1, '4': 2, '5': 3}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], 'rb') as f:
            data = pickle.load(f)

        patch = data['patch']
        gleason_score = data['gleason'][0]
        gleason_score = torch.tensor(self.label_map[gleason_score], dtype=torch.long)
        patch = Image.fromarray(patch)

        if self.transform:
            patch = self.transform(patch)

        return patch, gleason_score


def print_class_distribution(dataset, dataset_name="Dataset"):
    class_counts = Counter()

    for _, label in dataset:
        class_counts[int(label)] += 1  

    print(f"\nRozkład klas w {dataset_name}:")
    for cls, count in sorted(class_counts.items()):
        print(f"Klasa {cls}: {count} zdjęć")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


dataset = GleasonTilesLoader('/mnt/ip105/dpietrzak/panda/clear_paths.pkl', transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print_class_distribution(train_dataset, "Zbiór treningowy")
print_class_distribution(val_dataset, "Zbiór walidacyjny")
