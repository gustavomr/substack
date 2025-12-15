import numpy as np
import matplotlib.pyplot as plt

# Simple quadratic function: f(x,y) = x^2 + y^2
def f(x, y):
    return x**2 + y**2

def gradient(x, y):
    return np.array([2*x, 2*y])

def gradient_descent(start_x, start_y, learning_rate=0.1, max_iter=50):
    """Simple gradient descent with path tracking"""
    path = [[start_x, start_y]]
    x, y = start_x, start_y
    
    for i in range(max_iter):
        grad = gradient(x, y)
        x = x - learning_rate * grad[0]
        y = y - learning_rate * grad[1]
        path.append([x, y])
        
        # Stop if very close to minimum
        if abs(x) < 0.01 and abs(y) < 0.01:
            break
    
    return np.array(path)

# Create simple visualization
def visualize():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Create contour plot
    x_range = np.linspace(-3, 3, 50)
    y_range = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x_range, y_range)
    Z = f(X, Y)
    
    ax1.contour(X, Y, Z, levels=15, cmap='Blues')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Gradient Descent Path')
    ax1.grid(True, alpha=0.3)
    
    # Mark global minimum at (0,0)
    ax1.plot(0, 0, 'r*', markersize=15, label='Global Minimum (0,0)')
    
    # Test from different starting points
    starts = [(-2, -2), (2, 1), (-1, 2)]
    colors = ['red', 'blue', 'green']
    
    for i, (sx, sy) in enumerate(starts):
        path = gradient_descent(sx, sy)
        
        # Plot path
        ax1.plot(path[:, 0], path[:, 1], 'o-', color=colors[i], 
                markersize=4, linewidth=2, label=f'Start: ({sx},{sy})')
        
        # Plot loss over iterations
        losses = [f(p[0], p[1]) for p in path]
        ax2.plot(losses, color=colors[i], linewidth=2, 
                label=f'Start: ({sx},{sy})')
        
        print(f"Start ({sx},{sy}) → End ({path[-1][0]:.3f}, {path[-1][1]:.3f}) "
              f"Loss: {losses[-1]:.6f}")
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('f(x,y) = x² + y²')
    ax2.set_title('Loss Decrease')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Simple Gradient Descent Example")
    print("Function: f(x,y) = x² + y²")
    print("Global minimum at (0,0) with value 0")
    print("-" * 40)
    visualize()