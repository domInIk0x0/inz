import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
from PIL import Image
import numpy as np
from tqdm import tqdm



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
    def __init__(self, directory, transform=None):
        self.directory = directory
        with open(self.directory, 'rb') as f:
            self.files = pickle.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        patch = Image.fromarray(data['patch'])
        gleason_score = int(data['gleason'][0])
        image_id = data['image_id']

        if self.transform:
            patch = self.transform(patch)
        return patch, gleason_score, image_id, file_path



transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


dataset = GleasonTilesLoader('/mnt/ip105/dpietrzak/panda/clear_paths.pkl', transform=transform)
data_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18(num_classes=2, channels=1)
model.load_state_dict(torch.load('/mnt/ip105/dpietrzak/panda/model/resnet18_hel_vs_cancer_ADAMW_2.pth', map_location=device))
model.to(device)
model.eval()
print(len(data_loader))

features_list = []
labels_list = []
image_ids_list = []
paths_list = []

with torch.no_grad(): 
    for patches, labels, image_ids, paths in tqdm(data_loader, desc="Extracting features"):
        patches = patches.to(device)
        labels = labels.to(device)

        features = model.relu(model.layer4(model.layer3(model.layer2(model.layer1(model.maxpool(model.relu(model.batch_norm1(model.conv1(patches)))))))))
        features = model.avgpool(features)
        features = torch.flatten(features, 1).cpu().numpy() 

        features_list.append(features)
        labels_list.extend(labels.cpu().numpy())
        image_ids_list.extend(image_ids)
        paths_list.extend(paths)

features_array = np.concatenate(features_list, axis=0)
df = pd.DataFrame(features_array)
df['label'] = labels_list
df['image_id'] = image_ids_list
df['path'] = paths_list


output_path = '/mnt/ip105/dpietrzak/panda/extracted_features.pkl'
df.to_pickle(output_path)

