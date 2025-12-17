import math
import random
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


@dataclass
class Config:
    #Configs to create the dataset
    n_samples: int = 1_000
    noise: float = 0.1
    inner_radius: float = 0.5
    outer_radius: float = 1.0
    seed: int = 42

    #Configs to train the model
    batch_size: int = 64
    hidden_dim: int = 16
    lr: float = 0.05
    momentum: float = 0.9
    n_epochs: int = 100

    #Pytorch configs
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_circles(
    n_samples: int,
    inner_radius: float = 0.5,
    outer_radius: float = 1.0,
    noise: float = 0.1,
    plot: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a 2D dataset of two concentric circles.

    Class 0: points near the inner circle
    Class 1: points near the outer circle
    """
    n_inner = n_samples // 2
    n_outer = n_samples - n_inner

    # Inner circle
    theta_inner = 2 * math.pi * torch.rand(n_inner)
    r_inner = inner_radius + noise * torch.randn(n_inner)
    x_inner = torch.stack(
        [r_inner * torch.cos(theta_inner), r_inner * torch.sin(theta_inner)], dim=1
    )
    y_inner = torch.zeros(n_inner, dtype=torch.long)

    # Outer circle
    theta_outer = 2 * math.pi * torch.rand(n_outer)
    r_outer = outer_radius + noise * torch.randn(n_outer)
    x_outer = torch.stack(
        [r_outer * torch.cos(theta_outer), r_outer * torch.sin(theta_outer)], dim=1
    )
    y_outer = torch.ones(n_outer, dtype=torch.long)

    X = torch.cat([x_inner, x_outer], dim=0)
    y = torch.cat([y_inner, y_outer], dim=0)

    # Shuffle
    idx = torch.randperm(n_samples)

    # Plot the dataset using matplotlib (if requested)
    if plot:
        plt.figure(figsize=(5, 5))
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors="k", s=20)
        plt.title("Original circular dataset")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.axis("equal")
        plt.show()

    return X[idx], y[idx]


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 16, output_dim: int = 1):
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

    # Create dataset
    X, y = make_circles(
        n_samples=config.n_samples,
        inner_radius=config.inner_radius,
        outer_radius=config.outer_radius,
        noise=config.noise,
        plot=True,
    )

    # Train / validation split (80/20)
    n_train = int(0.8 * config.n_samples)
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]


    # Create data loaders with the bacth size defined in the config
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=config.batch_size,
        shuffle=False,
    )

    device = torch.device(config.device)

    model = SimpleMLP(hidden_dim=config.hidden_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config.lr, momentum=config.momentum
    )

    print(f"Using device: {device}")
    print(f"Training for {config.n_epochs} epochs...\n")

    for epoch in range(1, config.n_epochs + 1):

        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:

            #print how many iterations will be made
            #this is the number of batches
            #so if the batch size is 64 and the dataset has 1000 rows, then the number of iterations will be 1000/64 = 15.625
            #print(f"Iterations: {len(train_loader)}")

            #why I need this?
            #because the gradients are accumulated, so I need to zero them out
            #otherwise the gradients will be accumulated and the model will not learn
            optimizer.zero_grad()

            #why is this called logits?
            #because it is the output of the model before the sigmoid function
            #and the sigmoid function is used to convert the output to a probability
            #and the probability is used to calculate the loss
            #and the loss is used to update the weights of the model
            #and the weights are updated using the gradient descent algorithm
            logits = model(xb)
            probs = torch.sigmoid(logits)

            # why do we need to unsqueeze the yb?
            # because the yb is a 1D tensor, but the criterion expects a 2D tensor
            # so we need to unsqueeze it to make it a 2D tensor

            #why do we need to convert the yb to float?
            #because the criterion expects a float tensor
            #and the yb is a long tensor
            loss = criterion(probs, yb.float().unsqueeze(1))
        
            loss.backward()

            #why do we need to step the optimizer?
            #because the optimizer needs to update the weights of the model
            optimizer.step()

            #why loss.item()?
            #because the loss is a tensor, so we need to convert it to a float
            #and the loss.item() is used to get the value of the loss

            #why xb.size(0)?
            #this is the number of rows in the batch the could be different in the last batch

            #why we need to multiply the loss by the number of rows in the batch?
            #imagine that you have a dataset with 105 rows/samples and you are using a batch size of 50
            #the last batch will have 5 rows/samples If you just average the three loss values, the 5 samples in the last batch will have the same "weight" as the 50 samples in the first batch, which biases your results.
            running_loss += loss.item() * xb.size(0)


            preds = (probs > 0.5).long().squeeze(1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)

        train_loss = running_loss / total
        train_acc = correct / total


        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                probs = torch.sigmoid(logits)
                loss = criterion(probs, yb.float().unsqueeze(1))

                val_loss += loss.item() * xb.size(0)
                preds = (probs > 0.5).long().squeeze(1)
                val_correct += (preds == yb).sum().item()
                val_total += xb.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total


        if epoch == 1 or epoch % 10 == 0 or epoch == config.n_epochs:
            print(
                f"Epoch {epoch:3d}/{config.n_epochs} "
                f"Train loss: {train_loss:.4f}, acc: {train_acc:.3f} | "
                f"Val loss: {val_loss:.4f}, acc: {val_acc:.3f}"
            )

if __name__ == "__main__":
    cfg = Config()
    train(cfg)

