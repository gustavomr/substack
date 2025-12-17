import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
    
    # PyTorch configs
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # Visualize sample images
    print("Visualizing sample MNIST images...")
    num_examples = 6
    fig, axes = plt.subplots(1, num_examples, figsize=(15, 3))
    fig.suptitle('Sample MNIST Images', fontsize=14)
    
    for i in range(num_examples):
        # Get image from training set
        image, label = train_dataset[i]
        
        # Images are already in [0, 1] range, no denormalization needed
        img_vis = image.squeeze().numpy()
        
        # Plot image
        axes[i].imshow(img_vis, cmap='gray')
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    
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
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(test_dataset)}\n")

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
