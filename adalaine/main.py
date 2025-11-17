import torch
import matplotlib.pyplot as plt
import numpy as np

# --- 1. DATA AND HYPERPARAMETER SETUP ---#

# Linearly Separable Dataset AND example
X = torch.tensor([
    [0.0, 0.0],  # Point 1
    [0.0, 1.0],  # Point 2
    [1.0, 0.0],  # Point 3
    [1.0, 1.0]   # Point 4
])
T = torch.tensor([0.0, 0.0, 0.0, 1.0]) # Targets (Desired Outputs)

'''
# Linearly Separable Dataset
X = torch.tensor([
    [1.0, 1.0],  # Point 1
    [2.0, 3.0],  # Point 2
    [0.0, 0.0],  # Point 3
    [3.0, 0.5]   # Point 4
], dtype=torch.float32)
# Targets must be float32 for continuous error calculation
T = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32) # Targets (Desired Outputs)
'''

# Parameters
w = torch.tensor([0.0, 1.0], dtype=torch.float32)  # Initial weights
b = torch.tensor(-1.0, dtype=torch.float32)         # Initial bias
n = 0.01                    # ADALINE often needs a smaller learning rate (n) for stability
max_epochs = 1000           # Total iterations

print(f"Dataset X:\n{X}")
print(f"Targets T: {T}\n")
print("-" * 40)

# --- 2. ADALINE HELPER FUNCTION (Step is ONLY for prediction) --- #

def step_function(z):
    """Binary step: returns 1 if z > 0 else 0. Works for scalars and tensors."""
    ones = torch.ones_like(z, dtype=z.dtype)
    zeros = torch.zeros_like(z, dtype=z.dtype)
    return torch.where(z > 0, ones, zeros)

# --- 3. TRAINING LOOP (ADALINE using Delta Rule/LMS with SGD) --- #

all_mse_losses = []

for epoch in range(1, max_epochs + 1):
    current_errors = []
    
    # Iterate over EACH data point (Stochastic Gradient Descent)
    for i in range(len(X)):
        x_i = X[i]
        t_i = T[i]

        # 1. Forward Pass (Net Input Calculation)
        # ADALINE's linear activation means the output is the net input
        z = torch.dot(w, x_i) + b

        # 2. Activation Function (Linear Function / Identity Function)
        y = z
        
        # 3. Error Calculation (Continuous Error)
        # Error = output prediction - target
        error = y - t_i
        current_errors.append(error.item())

        # 4. Weight Adjustment (Delta Rule: updates EVERY time)
        # Delta_w = n * Error * x_i
        # Note: The derivative of 1/2*MSE is Error*x_i. We omit the 2 and 1/2
        # by absorbing them into the learning rate 'n'.
        
        delta_w = n * error * x_i
        delta_b = n * error 
        
        w = w - delta_w
        b = b - delta_b

    # Calculate Mean Squared Error (MSE) for the epoch
    mse = np.mean(np.array(current_errors)**2)
    all_mse_losses.append(mse)

    # 5. Reporting
    print(f"Epoch {epoch}: Mean Squared Error (MSE) = {mse:.6f}")

    # Optional Stopping Condition (e.g., if MSE is very low)
    if mse < 1e-4 and epoch > 10:
         print("\n*** CONVERGENCE ACHIEVED (Low MSE)! ***")
         break

# --- 4. FINAL RESULTS ---

print("\n" + "="*40)
print("FINAL RESULTS:")
print(f"Final Weights (w): {w}")
print(f"Final Bias (b): {b.item():.4f}")
print(f"Total Epochs: {epoch}")
print("Final MSE: {all_mse_losses[-1]:.6f}")
print("="*40)

# Test the final dataset
final_net_inputs = torch.matmul(X, w) + b
final_predictions = step_function(final_net_inputs)
print(f"Final Net Inputs (z): {final_net_inputs}")
print(f"Final Predictions:    {final_predictions}")
print(f"Original Targets:     {T}")

# --- 5. PLOT MSE LOSS CURVE ---
plt.figure(figsize=(7, 4))
plt.plot(range(1, len(all_mse_losses) + 1), all_mse_losses, marker='o')
plt.title('ADALINE (SGD) Training: Mean Squared Error Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# --- 6. PLOT WITH DECISION BOUNDARY ---
# (Using the same plotting code as the original Perceptron example)
with torch.no_grad():
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Plot data points
    for label, color in [(0.0, 'tab:blue'), (1.0, 'tab:orange')]:
        mask = T == label
        pts = X[mask]
        ax.scatter(pts[:, 0].tolist(), pts[:, 1].tolist(), label=f"Class {int(label)}", 
                   alpha=0.85, edgecolor='k', s=100, zorder=3)
    
    # Plot decision boundary: w[0]*x1 + w[1]*x2 + b = 0.5 (for {0, 1} targets)
    # Rearrange: x2 = (-w[0]*x1 - b + 0.5) / w[1]
    x1_min, x1_max = X[:, 0].min().item() - 0.5, X[:, 0].max().item() + 0.5
    x1_range = torch.linspace(x1_min, x1_max, 100)
    
    w0, w1, bias = w[0].item(), w[1].item(), b.item()
    
    if abs(w1) > 1e-6:
        # Note the slight shift to use the 0.5 threshold typical for {0,1} targets
        x2_boundary = (-w0 * x1_range - bias + 0.5) / w1 
        ax.plot(x1_range.tolist(), x2_boundary.tolist(), 'r--', linewidth=2, 
                label='Decision Boundary (z=0.5)', zorder=2)
    else:
        x1_vertical = (-bias + 0.5) / w0 if abs(w0) > 1e-6 else 0
        x2_min, x2_max = X[:, 1].min().item() - 0.5, X[:, 1].max().item() + 0.5
        ax.axvline(x=x1_vertical, color='r', linestyle='--', linewidth=2, 
                   label='Decision Boundary (z=0.5)', zorder=2)
    
    ax.set_title("ADALINE (SGD): Data Points and Learned Decision Boundary")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()