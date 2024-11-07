import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms.functional import to_pil_image
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pickle
from PIL import Image
import numpy as np
import os


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.i_downsample = i_downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.batch_norm3(self.conv3(x))
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        return self.relu(x)

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=4, channels=1):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, layers[0], out_channels=64)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
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
        return self.fc(x)

    def _make_layer(self, block, blocks, out_channels, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        layers = [block(self.in_channels, out_channels, downsample, stride)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    
def ResNet18(num_classes=4, channels=1):
    return ResNet(Bottleneck, [2, 2, 2, 2], num_classes=num_classes, channels=channels)


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
        patch = Image.fromarray(data['patch'])
        gleason_score = torch.tensor(self.label_map[data['gleason'][0]], dtype=torch.long)
        if self.transform:
            patch = self.transform(patch)
        return patch, gleason_score


def split_data(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    return random_split(dataset, [train_size, val_size, test_size])


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
    # Obliczanie metryk
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


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=100):
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
        
        scheduler.step()

        # Walidacja po każdej epoce
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

        
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# Przygotowanie zbiorów danych
dataset = GleasonTilesLoader('/mnt/ip105/dpietrzak/panda/clear_paths.pkl', transform=transform)
train_dataset, val_dataset, test_dataset = split_data(dataset)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18(num_classes=4, channels=1)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device)


evaluate_model(model, test_loader, device)
