import matplotlib.pyplot as plt
import torch

# Inputs (each column is a training example)
total_input = torch.tensor([[0, 0, 1, 1],
                            [0, 1, 0, 1]], dtype=torch.float32)

# XOR targets
y_xor = torch.tensor([[0, 1, 1, 0]], dtype=torch.float32)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


input_neurons, hidden_neurons, output_neurons = 2, 2, 1
samples = total_input.shape[1]
lr = 0.1
torch.manual_seed(42)

# Initialize weights (no bias, to mirror NumPy version)
w1 = torch.randn(hidden_neurons, input_neurons, requires_grad=True)
w2 = torch.randn(output_neurons, hidden_neurons, requires_grad=True)


def forward_prop(w1: torch.Tensor, w2: torch.Tensor, x: torch.Tensor):
    z1 = w1 @ x
    a1 = sigmoid(z1)
    z2 = w2 @ a1
    a2 = sigmoid(z2)
    return z1, a1, z2, a2


losses = []
epochs = 100_000

for _ in range(epochs):
    _, _, _, a2 = forward_prop(w1, w2, total_input)
    loss = (1 / (2 * samples)) * torch.sum((a2 - y_xor) ** 2)
    losses.append(loss.item())
    loss.backward()

    with torch.no_grad():
        w1 -= lr * w1.grad
        w2 -= lr * w2.grad

    w1.grad.zero_()
    w2.grad.zero_()


plt.plot(losses)
plt.xlabel("EPOCHS")
plt.ylabel("Loss value")
plt.show()


def predict(w1: torch.Tensor, w2: torch.Tensor, test_input: torch.Tensor):
    _, _, _, a2 = forward_prop(w1, w2, test_input)
    output = a2.squeeze().item()
    value = 1 if output >= 0.5 else 0
    inputs = [int(i) for i in test_input.squeeze().tolist()]
    print(f"For input {inputs} output is {value}")


test_cases = [torch.tensor([[0.0], [0.0]]),
              torch.tensor([[0.0], [1.0]]),
              torch.tensor([[1.0], [0.0]]),
              torch.tensor([[1.0], [1.0]])]

for test in test_cases:
    predict(w1, w2, test)