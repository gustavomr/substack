import torch

def pretty(tensor):
    """Return a plain Python list rounded to 6 decimal places for nice printing."""
    values = tensor.detach().cpu().numpy()
    return [[round(float(v), 6) for v in row] for row in values.reshape(1, -1)]


def main():
    torch.set_printoptions(precision=6, sci_mode=False)
    torch.set_num_threads(1)

    x = torch.tensor([[0.05, 0.10]], dtype=torch.float32)
    target = torch.tensor([[0.01, 0.99]], dtype=torch.float32)
    lr = 0.5

    w_input_hidden = torch.tensor(
        [[0.15, 0.25], [0.20, 0.30]], dtype=torch.float32, requires_grad=True
    )
    b_hidden = torch.tensor([0.35, 0.35], dtype=torch.float32, requires_grad=True)

    w_hidden_output = torch.tensor([[0.40, 0.50], [0.45, 0.55]], dtype=torch.float32, requires_grad=True)
    b_output = torch.tensor([0.60, 0.60], dtype=torch.float32, requires_grad=True)

    sigmoid = torch.nn.Sigmoid()

    print("== Setup ==")
    print(f"Inputs: x1={x[0,0]:.6f}, x2={x[0,1]:.6f}")
    print(f"Targets t1={target[0,0]:.6f}, t2={target[0,1]:.6f}, learning rate={lr}")

    # Forward calculations
    print("\n== Forward pass ==")
    z_hidden = x @ w_input_hidden + b_hidden
    for idx in range(z_hidden.shape[1]):
        w1 = w_input_hidden[0, idx].item()
        w2 = w_input_hidden[1, idx].item()
        bias = b_hidden[idx].item()
        weighted_sum = z_hidden[0, idx].item()
        print(
            f"h{idx+1}: z = x1*{w1:.6f} + x2*{w2:.6f} + b{idx+1}({bias:.6f}) = {weighted_sum:.6f}"
        )
    print(f"Hidden weighted sums [z1, z2] = {pretty(z_hidden)}")

    a_hidden = sigmoid(z_hidden)
    print(f"Hidden activations [a1, a2] = {pretty(a_hidden)}")

    z_output = a_hidden @ w_hidden_output + b_output
    for idx in range(z_output.shape[1]):
        w1 = w_hidden_output[0, idx].item()
        w2 = w_hidden_output[1, idx].item()
        bias = b_output[idx].item()
        weighted_sum = z_output[0, idx].item()
        print(
            f"Output {idx+1}: z{idx+3} = a1*{w1:.6f} + a2*{w2:.6f} + b{idx+3}({bias:.6f}) = {weighted_sum:.6f}"
        )
    print(f"Output weighted sums [z3, z4] = {pretty(z_output)}")

    output = sigmoid(z_output)
    print(f"Final outputs [y1, y2] = {pretty(output)}")

    loss = 0.5 * (target - output) ** 2
    total_loss = loss.sum()
    print(f"Loss per output (0.5 * (t - y)^2) = {pretty(loss)}")
    print(f"Total loss = {total_loss.item():.6f}")

    # Manual gradient walk-through
    print("\n== Manual gradient details ==")
    error_output = output - target
    print(f"Output errors (y - t) = {pretty(error_output)}")

    delta_output = error_output * output * (1 - output)
    print(f"Output deltas [δ3, δ4] = {pretty(delta_output)}")

    grad_w_hidden_output_manual = a_hidden.t() @ delta_output
    grad_b_output_manual = delta_output
    print(f"Grad hidden->output weights (manual) = {grad_w_hidden_output_manual.T}")
    print(f"Grad output biases [b3, b4] (manual) = {grad_b_output_manual}")

    delta_hidden = (delta_output @ w_hidden_output.t()) * a_hidden * (1 - a_hidden)
    grad_w_input_hidden_manual = x.t() @ delta_hidden
    grad_b_hidden_manual = delta_hidden
    print(f"Hidden deltas [δ1, δ2] = {delta_hidden}")
    print(f"Grad input->hidden weights (manual) = {grad_w_input_hidden_manual.T}")
    print(f"Grad hidden biases [b1, b2] (manual) = {grad_b_hidden_manual}")

    # Autograd confirmation
    print("\n== Autograd gradients ==")
    total_loss.backward()

    print("Input->hidden weights grad:")
    print(w_input_hidden.grad)
    print("Hidden biases grad:")
    print(b_hidden.grad)
    print("Hidden->output weights grad:")
    print(w_hidden_output.grad)
    print("Output biases grad:")
    print(b_output.grad)

    # Gradient descent update
    with torch.no_grad():
        w_input_hidden -= lr * w_input_hidden.grad
        b_hidden -= lr * b_hidden.grad
        w_hidden_output -= lr * w_hidden_output.grad
        b_output -= lr * b_output.grad

        w_input_hidden.grad.zero_()
        b_hidden.grad.zero_()
        w_hidden_output.grad.zero_()
        b_output.grad.zero_()

    print("\n== Updated parameters after 1 GD step ==")
    print("Input->hidden weights:")
    print(w_input_hidden)
    print("Hidden biases [b1, b2]:")
    print(b_hidden)
    print("Hidden->output weights:")
    print(w_hidden_output)
    print("Output biases [b3, b4]:")
    print(b_output)


if __name__ == "__main__":
    main()

