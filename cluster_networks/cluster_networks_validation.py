import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pickle
from PIL import Image

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

torch.multiprocessing.set_sharing_strategy('file_system')


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1, dropout_prob=0.4):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.i_downsample = i_downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.batch_norm2(self.conv2(x))
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        return self.relu(x)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=4, channels=1, dropout_prob=0.3):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], out_channels=64, dropout_prob=dropout_prob)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2, dropout_prob=dropout_prob)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2, dropout_prob=dropout_prob)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2, dropout_prob=dropout_prob)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

    def _make_layer(self, block, blocks, out_channels, stride=1, dropout_prob=0.4):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        layers = [block(self.in_channels, out_channels, downsample, stride, dropout_prob)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, dropout_prob=dropout_prob))
        return nn.Sequential(*layers)


def ResNet18(num_classes=4, channels=1, dropout_prob=0.3):
    return ResNet(Block, [2, 2, 2, 2], num_classes=num_classes, channels=channels, dropout_prob=dropout_prob)


class GleasonTilesLoader(Dataset):
    def __init__(self, path_to_pickle, transform=None):
        self.transform = transform
        self.label_map = {'0': 0, '3': 1, '4': 2, '5': 3}

        with open(path_to_pickle, 'rb') as f:
            self.files = pickle.load(f)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], 'rb') as f:
            data = pickle.load(f)

        patch = Image.fromarray(data['patch'])
        gleason_score = torch.tensor(self.label_map[data['gleason'][0]], dtype=torch.long)

        if self.transform:
            patch = self.transform(patch)
        return patch, gleason_score


def evaluate_model(model, data_loader, device):
    """
    Funkcja do oceny modelu na pełnym zbiorze.
    Wyświetla confusion matrix oraz accuracy, precision, recall, f1.
    """
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for patches, gleason_scores in data_loader:
            patches = patches.to(device)
            gleason_scores = gleason_scores.to(device)

            outputs = model(patches)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(gleason_scores.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print("\n=== Evaluation Results ===")
    print("Confusion Matrix:\n", cm)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ścieżki do plików z danymi (dla klastra 1 i 2)
    pictures_paths = [
        '/mnt/ip105/dpietrzak/panda/mniejszy_klaster_sciezki.pkl',
        '/mnt/ip105/dpietrzak/panda/wiekszy_klaster_sciezki.pkl'
    ]

    # Ścieżki do wytrenowanych parametrów (modele już po fine-tuningu)
    output_param_paths = [
        '/mnt/ip105/dpietrzak/panda/mniejszy_klaster_params_testowy.pth',
        '/mnt/ip105/dpietrzak/panda/wiekszy_klaster_params_testowy.pth'
    ]

    # Ta sama transformacja, co podczas trenowania
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # Analiza każdego z klastrów z osobna
    for i in range(len(pictures_paths)):
        print(f"\n=== Ewaluacja wytrenowanego modelu dla klastra nr {i+1} ===")

        # 1. Wczytanie całego datasetu dla konkretnego klastra (bez dzielenia na train/val/test).
        dataset = GleasonTilesLoader(pictures_paths[i], transform=transform)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

        # 2. Stworzenie obiektu modelu i wczytanie wytrenowanych parametrów
        model = ResNet18(num_classes=4, channels=1)
        model.load_state_dict(torch.load(output_param_paths[i], map_location=device))
        model.to(device)

        # 3. Ewaluacja na wszystkich danych tego klastra
        evaluate_model(model, data_loader, device)
