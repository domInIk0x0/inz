import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler

import pickle
from PIL import Image
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

torch.multiprocessing.set_sharing_strategy('file_system')

# ======================
# 1. Klasa Focal Loss
# ======================
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Implementacja Focal Loss:
    gamma   - siła tłumienia łatwych próbek
    alpha   - waga przypisana "ważniejszej" klasie
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # prawdopodobieństwo poprawnej klasy
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


def calculate_weights(dataset):
    class_counts = [0] * 4
    for _, label in dataset:
        class_counts[label.item()] += 1

    total_samples = sum(class_counts)
    weights = [total_samples / count if count > 0 else 0 for count in class_counts]

    print("Class Tile Counts:", class_counts)
    print("Class Weights:", weights)
    return torch.tensor(weights, dtype=torch.float)

# ==================================
# 2. Funkcja do WeightedRandomSampler
# ==================================
def make_weighted_sampler(train_dataset):
    """ Tworzy sampler, który losuje próbki zależnie od wagi danej klasy.
        W ten sposób w batchach klasy mniej liczne pojawiają się częściej.
    """
    class_counts = [0] * 4
    for _, label in train_dataset:
        class_counts[label.item()] += 1

    total_samples = len(train_dataset)
    # wagi per klasa
    weights_per_class = [0.0] * 4
    for i, c in enumerate(class_counts):
        if c != 0:
            weights_per_class[i] = total_samples / c

    sample_weights = [weights_per_class[label.item()] for _, label in train_dataset]
    sampler = WeightedRandomSampler(sample_weights, num_samples=total_samples, replacement=True)
    return sampler


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1, dropout_prob=0.3):
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

        self.layer1 = self._make_layer(block, layers[0], out_channels=64,  dropout_prob=dropout_prob)
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

    def _make_layer(self, block, blocks, out_channels, stride=1, dropout_prob=0.3):
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


def ResNet18(num_classes=4, channels=1, dropout_prob=0.4):
    return ResNet(Block, [2, 2, 2, 2], num_classes=num_classes, channels=channels, dropout_prob=dropout_prob)


class GleasonTilesLoader(Dataset):
    def __init__(self, path_to_pickle, transform=None):
        self.transform = transform
        self.label_map = {'0': 0, '3': 1, '4': 2, '5': 3}

        # Wczytanie listy plików z pliku .pkl
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


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, path, num_epochs=10):
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (patch, gleason_score) in enumerate(train_loader):
            patch, gleason_score = patch.to(device), gleason_score.to(device)

            optimizer.zero_grad()
            outputs = model(patch)
            loss = criterion(outputs, gleason_score)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Krok w schedulerze (zmiana LR)
        scheduler.step()

        # Walidacja
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for patch, gleason_score in val_loader:
                patch, gleason_score = patch.to(device), gleason_score.to(device)
                outputs = model(patch)
                _, predicted = torch.max(outputs, 1)
                val_total += gleason_score.size(0)
                val_correct += (predicted == gleason_score).sum().item()

        val_accuracy = 100 * val_correct / val_total
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Loss: {running_loss / len(train_loader):.4f}, "
              f"Val Acc: {val_accuracy:.2f}%")

    torch.save(model.state_dict(), path)


def evaluate_model(model, test_loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for patch, gleason_score in test_loader:
            patch, gleason_score = patch.to(device), gleason_score.to(device)
            outputs = model(patch)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(gleason_score.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print("Confusion Matrix:\n", cm)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")


# ----------------------------------------------
# Główna część skryptu - trenowanie (dostrajanie)
# ----------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pictures_paths = [
        '/mnt/ip105/dpietrzak/panda/wiekszy_klaster_sciezki.pkl',
        '/mnt/ip105/dpietrzak/panda/mniejszy_klaster_sciezki.pkl'
    ]

    output_param_paths = [
        '/mnt/ip105/dpietrzak/panda/wiekszy_klaster_params_testowy.pth',
        '/mnt/ip105/dpietrzak/panda/mniejszy_klaster_params_testowy.pth'
    ]

    # Ścieżka do modelu wstępnie wytrenowanego
    pretrained_model_path = '/mnt/ip105/dpietrzak/panda/model/resnet18_SGD_GRAY_2.pth'

    # Ustawienia
    num_epochs = 100
    batch_size = 128
    learning_rate = 0.001  # wikeszy LR
    use_focal_loss = True  # Przełącznik: True -> FocalLoss, False -> CrossEntropyLoss

    # Szersza augmentacja
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    for i in range(len(pictures_paths)):
        print(f"\n=== Dostrajanie modelu dla klastra nr {i+1} ===")

        dataset = GleasonTilesLoader(pictures_paths[i], transform=transform)

        train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
        train_size = int(train_ratio * len(dataset))
        val_size = int(val_ratio * len(dataset))
        test_size = len(dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size]
        )

        # --- Używam WeightedRandomSampler zamiast shuffle=True ---
        sampler = make_weighted_sampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=4)

        model = ResNet18(num_classes=4, channels=1, dropout_prob=0.4)


        model.to(device)

        # 5. Funkcja straty
        if use_focal_loss:
            criterion = FocalLoss(alpha=1.0, gamma=2.0)
        else:
            # Jeżeli wolisz zwykłe CrossEntropy z label smoothing:
            # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            # lub z wagami (bez samplera):
            # class_weights = calculate_weights(train_dataset)
            # criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
            criterion = nn.CrossEntropyLoss()

        # 6. Optymalizator i scheduler
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # Zamiast StepLR używam CosineAnnealingLR
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

        # 7. Trening
        train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            path=output_param_paths[i],
            num_epochs=num_epochs
        )

        # 8. Ocena
        evaluate_model(model, test_loader, device)
        print(f"Zakończono dostrajanie modelu nr {i+1}. Parametry zapisano w: {output_param_paths[i]}")
