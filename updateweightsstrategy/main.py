"""Compare SGD, mini-batch GD, and batch GD on a toy regression problem.

Run: python main.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter
import statistics

import matplotlib.pyplot as plt
import torch

# --- Config -----------------------------------------------------------------
SEED = 42
INPUT_DIM = 1
OUTPUT_DIM = 1
TRUE_WEIGHT = 3.0
TRUE_BIAS = -2.0
NUM_SAMPLES = 1024
NOISE_STD = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


torch.manual_seed(SEED)


def make_dataset(num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.linspace(-2.0, 2.0, steps=num_samples).unsqueeze(1)
    noise = torch.randn_like(x) * NOISE_STD
    y = TRUE_WEIGHT * x + TRUE_BIAS + noise
    return x.to(DEVICE), y.to(DEVICE)


@dataclass
class RunConfig:
    name: str
    batch_size: int
    learning_rate: float
    epochs: int


RUNS = [
    RunConfig(
        name="Batch GD (batch_size=all)",
        batch_size=NUM_SAMPLES,
        learning_rate=0.01,
        epochs=80,
    ),
       RunConfig(
        name="SGD (batch_size=1)",
        batch_size=1,
        learning_rate=0.01,
        epochs=80,
    ),
    RunConfig(
        name="Mini-batch (batch_size=32)",
        batch_size=32,
        learning_rate=0.01,
        epochs=80,
    ),

]


class LinearRegressor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(INPUT_DIM, OUTPUT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


mse_loss = torch.nn.MSELoss()


def train(config: RunConfig) -> tuple[list[float], float]:
    x, y = make_dataset(NUM_SAMPLES)
    model = LinearRegressor().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

    losses: list[float] = []
    num_batches = math.ceil(NUM_SAMPLES / config.batch_size)
    start_time = perf_counter()

    for epoch in range(1, config.epochs + 1):
        permutation = torch.randperm(NUM_SAMPLES, device=DEVICE)
        epoch_loss = 0.0

        for batch_idx in range(num_batches):
            start = batch_idx * config.batch_size
            end = min(start + config.batch_size, NUM_SAMPLES)
            batch_indices = permutation[start:end]
            x_batch, y_batch = x[batch_indices], y[batch_indices]

            optimizer.zero_grad()
            preds = model(x_batch)
            loss = mse_loss(preds, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * (end - start)

        epoch_loss /= NUM_SAMPLES
        losses.append(epoch_loss)

    final_w = model.linear.weight.item()
    final_b = model.linear.bias.item()
    elapsed = perf_counter() - start_time
    print(
        f"{config.name}: final loss {losses[-1]:.4f}, w={final_w:.2f} (true {TRUE_WEIGHT}), "
        f"b={final_b:.2f} (true {TRUE_BIAS}), time={elapsed*1000:.1f} ms"
    )
    return losses, elapsed

def plot_losses(named_losses: list[tuple[str, list[float]]]) -> None:
    if not named_losses:
        return
    plt.figure(figsize=(10, 5))
    for name, losses in named_losses:
        epochs = range(1, len(losses) + 1)
        plt.plot(epochs, losses, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training loss per epoch")
    plt.ylim(0, 2)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results = []
    for run in RUNS:
        losses, elapsed = train(run)
        results.append({
            "run": run,
            "losses": losses,
            "elapsed": elapsed,
        })

    print(
        "\nInterpretation:\n"
        "- Batch GD should show smaller initial drops but the lowest volatility (steady).\n"
        "- SGD should have the largest initial drop but the highest volatility (noisy).\n"
        "- Mini-batch should fall in between on both metrics."
    )

    plot_losses([(result["run"].name, result["losses"]) for result in results])
