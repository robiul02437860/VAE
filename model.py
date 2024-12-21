import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# Hyperparameters
latent_dim = 20
batch_size = 128
learning_rate = 1e-3
epochs = 200

# Data Preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

class Encoder(nn.Module):
    def __init__(self, input_dim= 784, hidden_dim= 400, latent_dim= 20):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_sigma =nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = nn.ReLU()(self.fc(x))
        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)
        return mu, sigma

# Decoder Network
class Decoder(nn.Module):
    def __init__(self, latent_dim=20, hidden_dim=400, output_dim=784):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = nn.ReLU()(self.fc1(z))
        x_recon = torch.sigmoid(self.fc2(h))
        return x_recon

# VAE Model
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, sigma):
        std = torch.exp(0.5*sigma)
        eps = torch.randn_like(std)
        return mu + std*eps
    
    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = self.reparameterize(mu, sigma)
        x_recons = self.decoder(z)
        return x_recons, mu, sigma

# Loss Function
def loss_function(x_recon, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div

# Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vae = VAE().to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

vae.train()
for epoch in range(epochs):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, 784).to(device)  # Flatten images
        optimizer.zero_grad()
        x_recon, mu, logvar = vae(data)
        loss = loss_function(x_recon, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset):.4f}")

# Visualize Original and Reconstructed Images
vae.eval()
with torch.no_grad():
    x, _ = next(iter(train_loader))  # Get a batch of data
    x = x.to(torch.float32)
    x = x.view(-1, 784).to(device)
    reconstruction, _, _ = vae(x)

    # Convert to 28x28 for visualization
    original_images = x.view(-1, 1, 28, 28)[:8]
    reconstructed_images = reconstruction.view(-1, 1, 28, 28)[:8]
    print(original_images.shape)
    print(reconstructed_images.shape)

    # Create a grid for visualization
    original_grid = make_grid(original_images, nrow=8, normalize=True)
    reconstructed_grid = make_grid(reconstructed_images, nrow=8, normalize=True)

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(original_grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    ax[0].set_title("Original Images")
    ax[0].axis("off")

    ax[1].imshow(reconstructed_grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    ax[1].set_title("Reconstructed Images")
    ax[1].axis("off")

    plt.show()

# Sampling and Generating Images
vae.eval()
with torch.no_grad():
    z = torch.randn(16, latent_dim).to(device)
    generated = vae.decoder(z).view(-1, 1, 28, 28).cpu()

# Visualizing Generated Images
import matplotlib.pyplot as plt
grid_img = torch.cat([torch.cat([generated[i * 4 + j] for j in range(4)], dim=2) for i in range(4)], dim=1)
plt.imshow(grid_img.squeeze(), cmap='gray')
plt.axis('off')
plt.show()
