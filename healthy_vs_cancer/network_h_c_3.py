import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pickle
from PIL import Image
import numpy as np
import os


# droput w blocku po batch_norm
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
        x = self.dropout(x)  # Dropout
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
        self.dropout = nn.Dropout(p=dropout_prob)  # Dropout przed FC
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


def split_data(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    return random_split(dataset, [train_size, val_size, test_size])


class GleasonTilesLoader(Dataset):
    def __init__(self, directory, transform=None, num_samples_per_class=None):
        """
        :param directory: Ścieżka do pliku pickle zawierającego ścieżki do plików PKL.
        :param transform: Transformacje do zastosowania na obrazach.
        :param num_samples_per_class: Liczba próbek do wybrania dla każdej klasy (równoważenie klas).
        """
        self.directory = directory
        with open(self.directory, 'rb') as f:
            all_files = pickle.load(f)

        self.class_healthy = [] 
        self.class_cancer = []  

        for file_path in all_files:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                gleason_score = data['gleason'][0]  
                if gleason_score == '0': 
                    self.class_healthy.append(file_path)
                else:  # Klasa rakowa
                    self.class_cancer.append((file_path, data['mask_sum']))

        
        self.class_cancer = sorted(self.class_cancer, key=lambda x: x[1], reverse=True)

        
        num_samples = num_samples_per_class or min(len(self.class_healthy), len(self.class_cancer))
        self.selected_healthy = self.class_healthy[:num_samples]
        self.selected_cancer = [x[0] for x in self.class_cancer[:num_samples]]

        
        self.files = self.selected_healthy + self.selected_cancer
        self.transform = transform
        self.label_map = {'0': 0, '3': 1, '4': 1, '5': 1}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        patch = Image.fromarray(data['patch'])
        gleason_score = torch.tensor(self.label_map[data['gleason'][0]], dtype=torch.long)

        if self.transform:
            patch = self.transform(patch)
        return patch, gleason_score


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


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, path, num_epochs=60):
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
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        torch.save(model.state_dict(), path)

        scheduler.step()


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

dataset = GleasonTilesLoader('/mnt/ip105/dpietrzak/panda/clear_paths.pkl', transform=transform)

train_dataset, val_dataset, test_dataset = split_data(dataset)

train_loader = DataLoader(train_dataset, batch_size=32, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

PATH = '/mnt/ip105/dpietrzak/panda/model/resnet18_hel_vs_cancer_ADAMW_2.pth'
model = ResNet18(num_classes=2, channels=1)

optimizer = optim.AdamW(model.parameters(), lr=0.002)


scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, path=PATH)
evaluate_model(model, test_loader, device)
