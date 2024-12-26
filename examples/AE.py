import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


def generate_distinct_shapes(num_samples=100, img_size=28):
    data = []
    labels = []

    for _ in range(num_samples):
        # Generowanie czarnych kwadratów z różnorodnością kształtu i intensywności
        black_square = np.zeros((img_size, img_size, 3))
        x1, y1 = np.random.randint(4, 12, size=2)  
        side = np.random.randint(8, 16)  
        intensity = np.random.uniform(0.3, 1.0)  
        for x in range(x1, x1 + side):
            for y in range(y1, y1 + side):
                if x < img_size and y < img_size:
                    black_square[x, y, :] = intensity  
   
        if np.random.rand() > 0.5:
            black_square[x1:x1 + side // 2, y1:y1 + side // 2, :] *= 0.5
        data.append(black_square)
        labels.append(0)  

        # Generowanie białych kółek z różnorodnością intensywności i wielkości
        white_circle = np.ones((img_size, img_size, 3))  
        cx, cy = np.random.randint(8, 20, size=2)  
        radius = np.random.randint(5, 12)  
        intensity = np.random.uniform(0.3, 1.0) 
        for x in range(img_size):
            for y in range(img_size):
                if (x - cx)**2 + (y - cy)**2 <= radius**2:
                    white_circle[x, y, :] *= intensity  

        if np.random.rand() > 0.5:
            cx_shift = np.random.randint(-2, 3)
            cy_shift = np.random.randint(-2, 3)
            for x in range(img_size):
                for y in range(img_size):
                    if (x - (cx + cx_shift))**2 + (y - (cy + cy_shift))**2 <= radius**2:
                        white_circle[x, y, :] *= 0.7
        data.append(white_circle)
        labels.append(1)  

    return np.array(data), np.array(labels)

data, labels = generate_distinct_shapes(500)
data = data.reshape(-1, 28 * 28 * 3)  

data_tensor = torch.tensor(data, dtype=torch.float32)
data_tensor /= 1.0  # Normalizacja do zakresu [0, 1]


class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28 * 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


model = SimpleAutoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    encoded, decoded = model(data_tensor)
    loss = criterion(decoded, data_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


model.eval()
with torch.no_grad():
    latent_space, reconstructed = model(data_tensor)
latent_space = latent_space.numpy()
reconstructed = reconstructed.numpy().reshape(-1, 28, 28, 3)
original = data_tensor.numpy().reshape(-1, 28, 28, 3)

# Wizualizacja przestrzeni latentnej
plt.figure(figsize=(8, 6))
scatter = plt.scatter(latent_space[:, 0], latent_space[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
plt.colorbar(scatter, label="Labels (0 = Black Squares, 1 = White Circles)")
plt.title("Latent Space Visualization")
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.show()

# Porównanie oryginalnych i zrekonstruowanych obrazów
def plot_original_and_reconstructed(original, reconstructed, labels, num_samples=5):
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, num_samples * 2))
    for i in range(num_samples):
        idx = np.random.choice(np.where(labels == 0)[0]) 
        axes[i, 0].imshow(original[idx])
        axes[i, 0].set_title("Original (Class 0)")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(reconstructed[idx])
        axes[i, 1].set_title("Reconstructed (Class 0)")
        axes[i, 1].axis("off")
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(num_samples, 2, figsize=(8, num_samples * 2))
    for i in range(num_samples):
        idx = np.random.choice(np.where(labels == 1)[0])  
        axes[i, 0].imshow(original[idx])
        axes[i, 0].set_title("Original (Class 1)")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(reconstructed[idx])
        axes[i, 1].set_title("Reconstructed (Class 1)")
        axes[i, 1].axis("off")
    plt.tight_layout()
    plt.show()


plot_original_and_reconstructed(original, reconstructed, labels)
