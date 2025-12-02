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
# 2. MODEL
# ----------------------------
class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ----------------------------
# 3. TRAINING / EVAL HELPERS
# ----------------------------
def l1_penalty(model: nn.Module) -> torch.Tensor:
    return torch.stack([param.abs().sum() for param in model.parameters()]).sum()


def l2_penalty(model: nn.Module) -> torch.Tensor:
    return torch.stack([param.pow(2).sum() for param in model.parameters()]).sum()


def train_model(epochs: int, l1_lambda: float = 0.0, l2_lambda: float = 0.0) -> float:
    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        if l1_lambda > 0:
            loss = loss + l1_lambda * l1_penalty(model)
        if l2_lambda > 0:
            loss = loss + l2_lambda * l2_penalty(model)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:04d} | loss={loss.item():.4f} | "
                f"L1={l1_lambda:.1e}, L2={l2_lambda:.1e}"
            )

    with torch.no_grad():
        preds = model(X_test).argmax(dim=1)
        accuracy = (preds == y_test).float().mean().item()
    return accuracy


def run_suite(epochs: int = 1000) -> None:
    configs = [
        ("No regularization", 0.0, 0.0),
        ("L2 (weight decay)", 0.0, 1e-3),
        ("L1 penalty", 1e-4, 0.0),
        ("Elastic Net", 5e-5, 5e-4),
    ]

    print("Training MLP with different regularization settings...\n")
    results: list[tuple[str, float]] = []
    for label, l1_lambda, l2_lambda in configs:
        print(f"--- {label} ---")
        acc = train_model(epochs=epochs, l1_lambda=l1_lambda, l2_lambda=l2_lambda)
        results.append((label, acc))
        print(f"Accuracy: {acc:.3f}\n")

    print("=== Summary ===")
    for label, acc in results:
        print(f"{label:20s}: {acc:.3f}")


if __name__ == "__main__":
    run_suite()

