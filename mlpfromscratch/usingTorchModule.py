import torch
import torch.nn as nn
import torch.optim as optim

# 1. Prepare the Data
# XOR Truth Table
# Input A | Input B | Output
# --------------------------
#    0    |    0    |    0
#    0    |    1    |    1
#    1    |    0    |    1
#    1    |    1    |    0

# Inputs: (4 samples, 2 features)
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)

# Targets: (4 samples, 1 label)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# 2. Define the Neural Network
class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        # Input Layer (2) -> Hidden Layer (5 neurons)
        self.hidden = nn.Linear(2, 5)
        # Hidden Layer (5) -> Output Layer (1 neuron)
        self.output = nn.Linear(5, 1)
        # Activation functions
        self.relu = nn.ReLU()      # Adds non-linearity
        self.sigmoid = nn.Sigmoid() # Squashes output between 0 and 1

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x

# Initialize model
model = XORModel()

# 3. Define Loss and Optimizer
criterion = nn.MSELoss() # Binary Cross Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.1) # Adam is usually faster than SGD

# 4. Training Loop
epochs = 20000

print("Starting training...")
for epoch in range(epochs):
    # Zero the gradients
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X)
    
    # Calculate loss
    loss = criterion(outputs, y)
    
    # Backward pass (backpropagation)
    loss.backward()
    
    # Update weights
    optimizer.step()
    
    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 5. Test the Model
print("\n--- Final Predictions ---")
with torch.no_grad():
    predictions = model(X)
    # Round the output to get 0 or 1
    predicted_labels = predictions.round()
    
    for i, input_sample in enumerate(X):
        print(f"Input: {input_sample.numpy()} -> Probability: {predictions[i].item():.4f} -> Prediction: {predicted_labels[i].item()}")