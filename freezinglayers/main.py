import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import os


class DeepNeuralNetwork(nn.Module):
    """Deep neural network with multiple hidden layers"""
    def __init__(self, input_dim: int = 20, hidden_dims: list = [128, 256, 512, 256, 128], num_classes: int = 2):
        super(DeepNeuralNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def create_dataset(n_samples=5000, n_features=20, n_classes=2, random_state=42):
    """Create a synthetic classification dataset"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 2,
        n_redundant=n_features // 4,
        n_classes=n_classes,
        random_state=random_state,
        n_clusters_per_class=1
    )
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    y_train = torch.LongTensor(y_train)
    y_val = torch.LongTensor(y_val)
    
    return X_train, X_val, y_train, y_val


def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, device='cpu'):
    """Train the model and return training history"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    print(f"Training on device: {device}")
    print(f"Training for {epochs} epochs...\n")
    
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs
    }


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create dataset
    print("Creating dataset...")
    X_train, X_val, y_train, y_val = create_dataset(
        n_samples=5000,
        n_features=20,
        n_classes=2
    )
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=64,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=64,
        shuffle=False
    )
    
    # Create model
    print("Creating deep neural network...")
    model = DeepNeuralNetwork(
        input_dim=20,
        hidden_dims=[128, 256, 512, 256, 128],
        num_classes=2
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")
    
    # Train model
    start_time = time.time()
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        lr=0.001,
        device=device
    )
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Final validation accuracy: {history['val_accs'][-1]:.4f}")
    
    # Save model
    model_path = 'saved_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': 20,
            'hidden_dims': [128, 256, 512, 256, 128],
            'num_classes': 2
        },
        'final_val_acc': history['val_accs'][-1],
        'training_time': training_time
    }, model_path)
    
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
