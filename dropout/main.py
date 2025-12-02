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
    def __init__(self, use_dropout: bool, p: float = 0.5):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(2, 32),
            nn.ReLU(),
        ]

        if use_dropout:
            layers.append(nn.Dropout(p))

        layers.extend(
            [
                nn.Linear(32, 32),
                nn.ReLU(),
            ]
        )

        if use_dropout:
            layers.append(nn.Dropout(p))

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
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | loss={loss.item():.4f}")

    with torch.no_grad():
        preds = model(X_test).argmax(dim=1)
        accuracy = (preds == y_test).float().mean().item()
    return accuracy


def run_experiment(use_dropout: bool) -> float:
    model = MLP(use_dropout)
    print("\nTraining", "with dropout" if use_dropout else "without dropout")
    return train_model(model)


# ----------------------------
# 4. RUN BOTH SETTINGS
# ----------------------------
baseline_acc = run_experiment(use_dropout=False)
dropout_acc = run_experiment(use_dropout=True)

print("\n=== Accuracy Comparison ===")
print(f"No Dropout   : {baseline_acc:.3f}")
print(f"With Dropout : {dropout_acc:.3f}")

