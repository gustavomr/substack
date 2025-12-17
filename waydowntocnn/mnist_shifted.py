import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import random
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class Config:
    # Configs to train the model
    batch_size: int = 64
    hidden_dim: int = 128
    lr: float = 0.01
    momentum: float = 0.9
    n_epochs: int = 10
    
    # Shift parameters
    shift_pixels: int = 4  # Number of pixels to shift (1D circular roll on flattened image)
    
    # PyTorch configs
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class ShiftedMNIST(Dataset):
    """Dataset wrapper that shifts pixels using 1D circular roll on flattened image"""
    def __init__(self, base_dataset, shift_pixels):
        self.base_dataset = base_dataset
        self.shift_pixels = shift_pixels
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        # 1D shift: roll the flattened image (circular shift)
        flat_image = image.view(-1)
        shifted_image = torch.roll(flat_image, self.shift_pixels)
        shifted_image = shifted_image.view(image.shape)
        return shifted_image, label


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 128, output_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train(config: Config) -> None:
    set_seed(config.seed)

    # Transform: convert to tensor (ToTensor already normalizes to [0, 1])
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load MNIST dataset
    train_dataset_base = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset_base = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # Training uses original images, validation uses shifted images
    train_dataset = train_dataset_base  # Original images for training
    test_dataset = ShiftedMNIST(test_dataset_base, config.shift_pixels)  # Shifted images for validation

    # Visualize original vs shifted images (using validation examples)
    print("Visualizing original vs shifted images (validation set)...")
    num_examples = 6
    fig, axes = plt.subplots(2, num_examples, figsize=(15, 5))
    fig.suptitle(f'Original vs Shifted Images (1D circular shift by {config.shift_pixels} pixels)', fontsize=14)
    
    for i in range(num_examples):
        # Get original image from validation set
        orig_image, label = test_dataset_base[i]
        # Get shifted image from validation set
        shifted_image, _ = test_dataset[i]
        
        # Images are already in [0, 1] range, no denormalization needed
        orig_vis = orig_image.squeeze().numpy()
        shifted_vis = shifted_image.squeeze().numpy()
        
        # Plot original
        axes[0, i].imshow(orig_vis, cmap='gray')
        axes[0, i].set_title(f'Original\nLabel: {label}')
        axes[0, i].axis('off')
        
        # Plot shifted
        axes[1, i].imshow(shifted_vis, cmap='gray')
        axes[1, i].set_title(f'Shifted (Val)\nLabel: {label}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    print()

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    
    val_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    device = torch.device(config.device)

    model = SimpleMLP(input_dim=784, hidden_dim=config.hidden_dim, output_dim=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config.lr, momentum=config.momentum
    )

    print(f"Using device: {device}")
    print(f"Training for {config.n_epochs} epochs...")
    print(f"Training samples: {len(train_dataset)} (original images)")
    print(f"Validation samples: {len(test_dataset)} (shifted images)")
    print(f"Validation images are 1D circular shifted by {config.shift_pixels} pixels\n")

    for epoch in range(1, config.n_epochs + 1):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.view(images.size(0), -1).to(device)  # Flatten to 784
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.view(images.size(0), -1).to(device)  # Flatten to 784
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total

        if epoch == 1 or epoch % 2 == 0 or epoch == config.n_epochs:
            print(
                f"Epoch {epoch:3d}/{config.n_epochs} "
                f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
                f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}"
            )


if __name__ == "__main__":
    cfg = Config()
    train(cfg)
