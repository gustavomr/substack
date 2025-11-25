import numpy as np
import matplotlib.pyplot as plt

# 1. Define the Function (The "Valley")
# We want to find the value of x that minimizes y = x^2
def cost_function(x):
    return x**2

# The gradient tells us the slope (direction and steepness)
def gradient(x):
    return 2*x

# 2. Gradient Descent Algorithm
def gradient_descent(start_x, learning_rate, n_iterations):
    # Track history for plotting
    x_path = [start_x]
    y_path = [cost_function(start_x)]
    
    x = start_x
    for _ in range(n_iterations):
        grad = gradient(x)
        # The Core Learning Step:
        # new_position = old_position - (step_size * slope)
        x = x - learning_rate * grad 
        
        x_path.append(x)
        y_path.append(cost_function(x))
        
    return x_path, y_path

# 3. Visualize Different Learning Rates
learning_rates = [0.01, 0.1, 0.95, 1.05]
start_x = -4  # Start on the left side of the valley
iterations = 10

# Setup Plots
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
x_bg = np.linspace(-5, 5, 100)
y_bg = cost_function(x_bg)

for i, lr in enumerate(learning_rates):
    # Run the training simulation
    x_steps, y_steps = gradient_descent(start_x, lr, iterations)
    
    # Plot the background curve
    axes[i].plot(x_bg, y_bg, color='lightgray', label='Error Surface')
    
    # Plot the steps taken by the model
    axes[i].plot(x_steps, y_steps, color='red', marker='o', linestyle='--', label='Steps')
    
    # Titles and formatting
    axes[i].set_title(f'Learning Rate: {lr}')
    axes[i].set_ylim(0, 27)
    if i == 0: axes[i].legend()

plt.show()