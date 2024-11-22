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


def calculate_weights(dataset):
    class_counts = [0] * 4
    for _, label in dataset:
        class_counts[label.item()] += 1

    total_samples = sum(class_counts)
    weights = [total_samples / count if count > 0 else 0 for count in class_counts]

    print("Class Tile Counts:", class_counts)
    print("Class Weights:", weights)
    return torch.tensor(weights, dtype=torch.float)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.i_downsample = i_downsample
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.dropout(x)
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
        self.fc = nn.Sequential(
            nn.Linear(512 * block.expansion, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

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

    def _make_layer(self, block, num_blocks, out_channels, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        layers = [block(self.in_channels, out_channels, downsample, stride)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)


def ResNet50(num_classes=4, channels=1):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, channels=channels)


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


def train_model_with_early_stopping(
    model, train_loader, val_loader, criterion, optimizer, scheduler, device, path, num_epochs=50, patience=5
):
    model.to(device)
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for patch, gleason_score in train_loader:
            patch, gleason_score = patch.to(device), gleason_score.to(device)
            optimizer.zero_grad()
            outputs = model(patch)
            loss = criterion(outputs, gleason_score)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for patch, gleason_score in val_loader:
                patch, gleason_score = patch.to(device), gleason_score.to(device)
                outputs = model(patch)
                loss = criterion(outputs, gleason_score)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += gleason_score.size(0)
                val_correct += (predicted == gleason_score).sum().item()

        val_accuracy = 100 * val_correct / val_total
        val_loss /= len(val_loader)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, "
            f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break


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


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

dataset = GleasonTilesLoader('/mnt/ip105/dpietrzak/panda/clear_paths.pkl', transform=transform)
train_dataset, val_dataset, test_dataset = split_data(dataset)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = calculate_weights(train_dataset)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

PATH = '/mnt/ip105/dpietrzak/panda/model/resnet50_SGD_GRAY_earlystopping.pth'
model = ResNet50(num_classes=4, channels=1)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

train_model_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, scheduler, device, path=PATH)

model.load_state_dict(torch.load(PATH))
evaluate_model(model, test_loader, device)