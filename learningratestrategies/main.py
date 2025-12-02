"""Compare learning-rate schedulers and optimizers on Kaggle's Titanic dataset."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_LR = 0.05
EPOCHS = 100
BATCH_SIZE = 16
VAL_SPLIT = 0.2
DATA_DIR = Path(__file__).resolve().parent / "data" / "titanic"
TRAIN_CSV = DATA_DIR / "train.csv"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def load_titanic() -> tuple[torch.Tensor, torch.Tensor]:
    """Load & preprocess the Kaggle Titanic dataset for binary classification."""

    if not TRAIN_CSV.exists():
        raise FileNotFoundError(
            "Missing Titanic data. Download via `kaggle competitions download -c titanic` "
            f"into {DATA_DIR} and unzip train.csv before running."
        )

    df = pd.read_csv(TRAIN_CSV)
    target = torch.tensor(df["Survived"].values, dtype=torch.long)

    num_cols = ["Age", "SibSp", "Parch", "Fare"]
    cat_cols = ["Pclass", "Sex", "Embarked"]

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna("missing")

    numeric = df[num_cols]
    numeric = (numeric - numeric.mean()) / numeric.std()
    categoricals = pd.get_dummies(df[cat_cols], drop_first=False)
    features = pd.concat([numeric, categoricals], axis=1).astype("float32")

    inputs = torch.tensor(features.values, dtype=torch.float32)
    return inputs.to(DEVICE), target.to(DEVICE)


class SimpleClassifier(torch.nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


@dataclass(frozen=True)
class LRStrategy:
    name: str
    scheduler_fn: Callable[[torch.optim.Optimizer, int], torch.optim.lr_scheduler._LRScheduler | None]
    step_policy: str  # "none", "batch", "epoch", "plateau"


@dataclass(frozen=True)
class OptimizerStrategy:
    name: str
    builder: Callable[[torch.nn.Module], torch.optim.Optimizer]
    scheduler_fn: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler | None] = lambda optimizer: None


def _build_strategy(name: str) -> LRStrategy:
    def scheduler_factory(optimizer: torch.optim.Optimizer, steps_per_epoch: int):
        if name == "constant":
            return None
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
        if name == "exp":
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        if name == "cyclical":
            step_up = max(1, steps_per_epoch // 50)
            return torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=BASE_LR / 10,
                max_lr=BASE_LR * 5,
                step_size_up=step_up,
                cycle_momentum=False,
            )
        if name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=5, factor=0.5, threshold=1e-3
            )
        raise ValueError(f"Unknown strategy: {name}")

    step_policy = {
        "constant": "none",
        "step": "epoch",
        "exp": "epoch",
        "cosine": "epoch",
        "cyclical": "batch",
        "plateau": "plateau",
    }[name]

    label = {
        "constant": "Constant LR",
        "step": "Step Decay",
        "exp": "Exponential Decay",
        "cosine": "Cosine Annealing",
        "cyclical": "Cyclical (triangular)",
        "plateau": "ReduceLROnPlateau",
    }[name]

    return LRStrategy(label, scheduler_factory, step_policy)


STRATEGIES: tuple[LRStrategy, ...] = tuple(
    _build_strategy(name)
    for name in ("constant", "step", "exp", "cosine", "cyclical", "plateau")
)


def _default_optimizer_scheduler(step_size: int = 50, gamma: float = 0.9):
    return lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


OPTIMIZERS: tuple[OptimizerStrategy, ...] = (
    OptimizerStrategy(
        "Adagrad",
        lambda model: torch.optim.Adagrad(model.parameters(), lr=0.05),
        _default_optimizer_scheduler(step_size=75, gamma=0.85),
    ),
    OptimizerStrategy(
        "RMSprop",
        lambda model: torch.optim.RMSprop(model.parameters(), lr=0.01),
        _default_optimizer_scheduler(step_size=60, gamma=0.92),
    ),
    OptimizerStrategy(
        "Adam",
        lambda model: torch.optim.Adam(model.parameters(), lr=0.01),
        _default_optimizer_scheduler(step_size=80, gamma=0.9),
    ),
    OptimizerStrategy(
        "Adadelta",
        lambda model: torch.optim.Adadelta(model.parameters(), lr=1.0),
        _default_optimizer_scheduler(step_size=40, gamma=0.7),
    ),
)


def evaluate(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, criterion: torch.nn.Module
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            count += batch_x.size(0)
    return total_loss / count, correct / count


def train(
    strategy: LRStrategy,
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> tuple[list[float], list[float], list[float]]:
    set_seed()
    dataset = torch.utils.data.TensorDataset(features, labels)
    val_size = max(1, int(len(dataset) * VAL_SPLIT))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE)

    model = SimpleClassifier(features.size(1), num_classes).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=BASE_LR)
    scheduler = strategy.scheduler_fn(optimizer, len(train_loader))
    criterion = torch.nn.CrossEntropyLoss()

    train_losses: list[float] = []
    val_accuracies: list[float] = []
    lr_history: list[float] = []
    batch_scheduler = strategy.step_policy == "batch"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        count = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            count += batch_x.size(0)
            if batch_scheduler:
                lr_history.append(optimizer.param_groups[0]["lr"])

            if scheduler and batch_scheduler:
                scheduler.step()

        train_loss = running_loss / count
        train_losses.append(train_loss)

        if scheduler:
            if strategy.step_policy == "epoch":
                scheduler.step()
            elif strategy.step_policy == "plateau":
                scheduler.step(train_loss)

        if not batch_scheduler:
            lr_history.append(optimizer.param_groups[0]["lr"])

        val_loss, val_acc = evaluate(model, val_loader, criterion)
        val_accuracies.append(val_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"[{strategy.name}] Epoch {epoch:03d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
            )

    return train_losses, lr_history, val_accuracies


def train_with_optimizer(
    opt_strategy: OptimizerStrategy,
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> tuple[list[float], list[float], list[float]]:
    set_seed()
    dataset = torch.utils.data.TensorDataset(features, labels)
    val_size = max(1, int(len(dataset) * VAL_SPLIT))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE)

    model = SimpleClassifier(features.size(1), num_classes).to(DEVICE)
    optimizer = opt_strategy.builder(model)
    scheduler = opt_strategy.scheduler_fn(optimizer)
    criterion = torch.nn.CrossEntropyLoss()

    train_losses: list[float] = []
    val_accuracies: list[float] = []
    lr_history: list[float] = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            sample_count += batch_x.size(0)

        train_losses.append(running_loss / max(1, sample_count))

        val_loss, val_acc = evaluate(model, val_loader, criterion)
        val_accuracies.append(val_acc)
        lr_history.append(optimizer.param_groups[0]["lr"])

        if scheduler:
            scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"[{opt_strategy.name}] Epoch {epoch:03d}: train_loss={train_losses[-1]:.4f}, val_loss={val_loss:.4f}"
            )

    return train_losses, lr_history, val_accuracies


def summarize_schedulers(histories: dict[str, tuple[list[float], list[float], list[float]]]) -> None:
    cols = 3
    rows = math.ceil(len(histories) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    for ax, (name, (losses, lrs, accs)) in zip(axes, histories.items()):
        ax.plot(losses, label="Train Loss", color="tab:blue")
        ax.plot(accs, label="Val Accuracy", color="tab:green", linestyle="--")
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss / Accuracy")
        ax.set_xlim(0, EPOCHS)
        ax2 = ax.twinx()
        ax2.plot(lrs, label="LR", color="tab:orange", alpha=0.6)
        ax2.set_ylabel("Learning Rate")
        ax.grid(True, linestyle="--", alpha=0.3)

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper right", fontsize="small")

    for ax in axes[len(histories) :]:
        ax.axis("off")

    fig.suptitle("Learning-Rate Schedulers on Titanic Classification", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()


def summarize_optimizers(histories: dict[str, tuple[list[float], list[float], list[float]]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for ax, (name, (losses, lrs, accs)) in zip(axes, histories.items()):
        ax.plot(losses, label="Train Loss", color="tab:blue")
        ax.plot(accs, label="Val Accuracy", color="tab:green", linestyle="--")
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_xlim(0, EPOCHS)
        ax.set_ylabel("Loss / Accuracy")
        ax2 = ax.twinx()
        ax2.plot(lrs, label="LR", color="tab:orange", alpha=0.6)
        ax2.set_ylabel("Learning Rate")
        ax.grid(True, linestyle="--", alpha=0.3)

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper right", fontsize="small")

    for ax in axes[len(histories) :]:
        ax.axis("off")

    fig.suptitle("Optimizer Comparison on Titanic Classification", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()


def main() -> None:
    features, labels = load_titanic()
    num_classes = int(labels.unique().numel())

    scheduler_histories: dict[str, tuple[list[float], list[float], list[float]]] = {}
    for strategy in STRATEGIES:
        print(f"Training with {strategy.name}...")
        scheduler_histories[strategy.name] = train(strategy, features, labels, num_classes)
    summarize_schedulers(scheduler_histories)

    optimizer_histories: dict[str, tuple[list[float], list[float], list[float]]] = {}
    for optimizer_strategy in OPTIMIZERS:
        print(f"Training with optimizer {optimizer_strategy.name}...")
        optimizer_histories[optimizer_strategy.name] = train_with_optimizer(
            optimizer_strategy, features, labels, num_classes
        )
    summarize_optimizers(optimizer_histories)


if __name__ == "__main__":
    main()
