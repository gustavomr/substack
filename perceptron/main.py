import torch
import matplotlib.pyplot as plt

# --- 1. DATA AND HYPERPARAMETER SETUP --- #

# Linearly Separable Dataset - AND logical example
X = torch.tensor([
    [0.0, 0.0],  # Point 1
    [0.0, 1.0],  # Point 2
    [1.0, 0.0],  # Point 3
    [1.0, 1.0]   # Point 4
])
T = torch.tensor([0.0, 0.0, 0.0, 1.0]) # Targets (Desired Outputs)

# Parameters
w = torch.tensor([0.0, 1.0])  # Initial weights
b = torch.tensor(-1.0)        # Initial bias
n = 1.0                       # Learning rate
max_epochs = 100              # Safety limit to prevent infinite loops

'''
# Linearly Separable Dataset
X = torch.tensor([
    [1.0, 1.0],  # Point 1
    [2.0, 3.0],  # Point 2
    [0.0, 0.0],  # Point 3
    [3.0, 0.5]   # Point 4
])
T = torch.tensor([1.0, 1.0, 0.0, 0.0]) # Targets (Desired Outputs)

# Parameters
w = torch.tensor([0.1, 0.1])  # Initial weights
b = torch.tensor(0.0)         # Initial bias
n = 0.1                       # Learning rate
max_epochs = 100              # Safety limit to prevent infinite loops '''


print(f"Dataset X:\n{X}")
print(f"Targets T: {T}\n")
print("-" * 40)

# --- Quick plot of datapoints colored by target (0/1) --- #
with torch.no_grad():
    fig, ax = plt.subplots(figsize=(5, 4))
    for label, color in [(0.0, 'tab:blue'), (1.0, 'tab:orange')]:
        mask = T == label
        pts = X[mask]
        ax.scatter(pts[:, 0].tolist(), pts[:, 1].tolist(), label=f"Class {int(label)}", alpha=0.85, edgecolor='k')
    ax.set_title("Dataset Points by Target")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

# --- 2. PERCEPTRON HELPER FUNCTION --- #

def step_function(z):
    """Binary step: returns 1 if z > 0 else 0. Works for scalars and tensors."""
    ones = torch.ones_like(z, dtype=z.dtype)
    zeros = torch.zeros_like(z, dtype=z.dtype)
    return torch.where(z > 0, ones, zeros)

# --- 3. TRAINING LOOP --- #

for epoch in range(1, max_epochs + 1):
    misclassifications = 0
    print(f"\n--- Epoch {epoch} start ---")
    print(f"Starting weights: w={w.tolist()}, b={b.item():.6f}")
    
    # Iterate over EACH data point (Stochastic/Online Learning)
    for i in range(len(X)):
        x_i = X[i] # Input vector
        t_i = T[i] # Target vector

        # 1. Forward Pass (Weighted sum of inputs + bias)
        z = torch.dot(w, x_i) + b
        
        # 2. Output prediction(Step function activation)
        y = step_function(z)

        # 3. Error Calculation (Target - Output prediction)
        error = t_i - y

        # Debug prints for current sample calculations
        print(f"  Sample {i+1}:")
        print(f"    x_i = {x_i.tolist()}, t_i = {t_i.item():.1f}")
        print(f"    z = w·x_i + b = {z.item():.6f}")
        print(f"    y = step(z) = {y.item():.1f}")
        print(f"    error = t_i - y = {error.item():.1f}")

        # 4. Weight Adjustment (ONLY if error occurs)
        if error.item() != 0:
            misclassifications += 1
            
            # Perceptron Update Rule
            delta_w = n * error * x_i
            delta_b = n * error
            
            w = w + delta_w
            b = b + delta_b
            print(f"    update -> Δw = {delta_w.tolist()}, Δb = {delta_b.item():.1f}")
            print(f"    updated -> w = {w.tolist()}, b = {b.item():.6f}")
        else:
            print("    no update (correct classification)")

    print(f"Epoch {epoch}: {misclassifications} misclassifications.")

    # Stopping Condition: Convergence or max_epochs reached
    if misclassifications == 0:
        print("\n*** CONVERGENCE ACHIEVED! ***")
        break

    if epoch == max_epochs:
        print("\n*** Maximum epoch limit reached. ***")

# --- 4. FINAL RESULTS ---

print("\n" + "="*40)
print("FINAL RESULTS:")
print(f"Final Weights (w): {w}")
print(f"Final Bias (b): {b.item():.4f}")
print(f"Total Epochs: {epoch}")
print("="*40)

# Test the final dataset
final_predictions = step_function(torch.matmul(X, w) + b)
print(f"Final Predictions: {final_predictions}")
print(f"Original Targets:  {T}")

# --- 5. PLOT WITH DECISION BOUNDARY ---
with torch.no_grad():
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Plot data points
    for label, color in [(0.0, 'tab:blue'), (1.0, 'tab:orange')]:
        mask = T == label
        pts = X[mask]
        ax.scatter(pts[:, 0].tolist(), pts[:, 1].tolist(), label=f"Class {int(label)}", 
                   alpha=0.85, edgecolor='k', s=100, zorder=3)
    
    # Plot decision boundary: w[0]*x1 + w[1]*x2 + b = 0
    # Rearrange: x2 = (-w[0]*x1 - b) / w[1]
    x1_min, x1_max = X[:, 0].min().item() - 0.5, X[:, 0].max().item() + 0.5
    x1_range = torch.linspace(x1_min, x1_max, 100)
    
    w0, w1, bias = w[0].item(), w[1].item(), b.item()
    
    if abs(w1) > 1e-6:  # Normal case: non-vertical line
        x2_boundary = (-w0 * x1_range - bias) / w1
        ax.plot(x1_range.tolist(), x2_boundary.tolist(), 'r--', linewidth=2, 
                label='Decision Boundary', zorder=2)
    else:  # Vertical line: w1 ≈ 0
        x1_vertical = -bias / w0 if abs(w0) > 1e-6 else 0
        x2_min, x2_max = X[:, 1].min().item() - 0.5, X[:, 1].max().item() + 0.5
        ax.axvline(x=x1_vertical, color='r', linestyle='--', linewidth=2, 
                   label='Decision Boundary', zorder=2)
    
    ax.set_title("Perceptron: Data Points and Learned Decision Boundary")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()