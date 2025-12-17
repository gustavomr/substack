import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


# ----------------------------
# 1. DATA
# ----------------------------
X, y = make_moons(noise=0.2, random_state=42)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# ----------------------------
# 2. MODELS
# ----------------------------
class MLP(nn.Module):
    def __init__(self, use_batchnorm: bool):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(2, 32),
        ]

        if use_batchnorm:
            layers.append(nn.BatchNorm1d(32))

        layers.append(nn.ReLU())

        layers.extend(
            [
                nn.Linear(32, 32),
            ]
        )

        if use_batchnorm:
            layers.append(nn.BatchNorm1d(32))

        layers.append(nn.ReLU())
        layers.append(nn.Linear(32, 2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ----------------------------
# 3. TRAINING / EVAL HELPERS
# ----------------------------
def train_model(model: nn.Module, epochs: int = 1000) -> float:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, epochs + 1):
        # Set model to training mode (important for BatchNorm)
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | loss={loss.item():.4f}")

    # Set model to evaluation mode (important for BatchNorm)
    model.eval()
    with torch.no_grad():
        preds = model(X_test).argmax(dim=1)
        accuracy = (preds == y_test).float().mean().item()
    return accuracy


def run_experiment(use_batchnorm: bool) -> float:
    model = MLP(use_batchnorm)
    print("\nTraining", "with batch normalization" if use_batchnorm else "without batch normalization")
    return train_model(model)


# ----------------------------
# 4. RUN BOTH SETTINGS
# ----------------------------
baseline_acc = run_experiment(use_batchnorm=False)
batchnorm_acc = run_experiment(use_batchnorm=True)

print("\n=== Accuracy Comparison ===")
print(f"No BatchNorm   : {baseline_acc:.3f}")
print(f"With BatchNorm : {batchnorm_acc:.3f}")
