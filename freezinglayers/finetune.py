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
    
    def freeze_all_except_last_two(self):
        """Freeze all layers except the last two Linear layers and their associated BatchNorm layers"""
        # Freeze all parameters first
        for param in self.parameters():
            param.requires_grad = False
        
        # Get all Linear and BatchNorm layers in order
        # The network structure is: Linear -> BN -> ReLU -> Dropout -> Linear -> BN -> ReLU -> Dropout -> ... -> Linear
        linear_layers = []
        bn_layers = []
        
        for module in self.network:
            if isinstance(module, nn.Linear):
                linear_layers.append(module)
            elif isinstance(module, nn.BatchNorm1d):
                bn_layers.append(module)
        
        # Unfreeze the last two Linear layers
        if len(linear_layers) >= 2:
            # Last Linear layer (output layer)
            linear_layers[-1].weight.requires_grad = True
            linear_layers[-1].bias.requires_grad = True
            
            # Second-to-last Linear layer
            linear_layers[-2].weight.requires_grad = True
            linear_layers[-2].bias.requires_grad = True
            
            # Unfreeze the BatchNorm layer associated with the second-to-last Linear layer
            # (The BatchNorm comes right after the Linear layer in our structure)
            if len(bn_layers) >= len(linear_layers) - 1:
                # The BatchNorm before the last Linear layer
                bn_layers[-1].weight.requires_grad = True
                bn_layers[-1].bias.requires_grad = True
        
        # Count frozen and unfrozen parameters
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return frozen_params, trainable_params
    
    def unfreeze_all(self):
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True


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


def train_model(model, train_loader, val_loader, epochs=30, lr=0.001, device='cpu'):
    """Train the model and return training history and time"""
    criterion = nn.CrossEntropyLoss()
    
    # Only optimize parameters that require gradients
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    start_time = time.time()
    
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
    
    training_time = time.time() - start_time
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'training_time': training_time
    }


def main():
    # Check if model exists
    model_path = 'saved_model.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please run main.py first to train and save the model.")
        return
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load saved model
    print("Loading saved model...")
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['model_config']
    
    model = DeepNeuralNetwork(
        input_dim=model_config['input_dim'],
        hidden_dims=model_config['hidden_dims'],
        num_classes=model_config['num_classes']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded. Original validation accuracy: {checkpoint['final_val_acc']:.4f}\n")
    
    # Create dataset (using same parameters as training)
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
    
    print("=" * 80)
    print("EXPERIMENT 1: Fine-tuning with FROZEN layers (only last 2 layers trainable)")
    print("=" * 80)
    
    # Create a copy of the model for frozen fine-tuning
    model_frozen = DeepNeuralNetwork(
        input_dim=model_config['input_dim'],
        hidden_dims=model_config['hidden_dims'],
        num_classes=model_config['num_classes']
    ).to(device)
    model_frozen.load_state_dict(checkpoint['model_state_dict'])
    
    # Freeze all except last two layers
    frozen_params, trainable_params = model_frozen.freeze_all_except_last_two()
    total_params = frozen_params + trainable_params
    
    print(f"\nFrozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print()
    
    # Fine-tune with frozen layers
    history_frozen = train_model(
        model=model_frozen,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=30,
        lr=0.001,
        device=device
    )
    
    frozen_time = history_frozen['training_time']
    frozen_final_acc = history_frozen['val_accs'][-1]
    
    print(f"\nFrozen fine-tuning completed in {frozen_time:.2f} seconds")
    print(f"Final validation accuracy: {frozen_final_acc:.4f}")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Fine-tuning with ALL layers UNFROZEN (full fine-tuning)")
    print("=" * 80)
    
    # Create a copy of the model for full fine-tuning
    model_unfrozen = DeepNeuralNetwork(
        input_dim=model_config['input_dim'],
        hidden_dims=model_config['hidden_dims'],
        num_classes=model_config['num_classes']
    ).to(device)
    model_unfrozen.load_state_dict(checkpoint['model_state_dict'])
    model_unfrozen.unfreeze_all()
    
    trainable_params_all = sum(p.numel() for p in model_unfrozen.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {trainable_params_all:,} (100%)\n")
    
    # Fine-tune with all layers unfrozen
    history_unfrozen = train_model(
        model=model_unfrozen,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=30,
        lr=0.001,
        device=device
    )
    
    unfrozen_time = history_unfrozen['training_time']
    unfrozen_final_acc = history_unfrozen['val_accs'][-1]
    
    print(f"\nFull fine-tuning completed in {unfrozen_time:.2f} seconds")
    print(f"Final validation accuracy: {unfrozen_final_acc:.4f}")
    
    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print(f"{'Metric':<30} {'Frozen (Last 2)':<20} {'Unfrozen (All)':<20} {'Difference':<20}")
    print("-" * 80)
    print(f"{'Training Time (seconds)':<30} {frozen_time:<20.2f} {unfrozen_time:<20.2f} {unfrozen_time - frozen_time:<20.2f}")
    print(f"{'Final Validation Accuracy':<30} {frozen_final_acc:<20.4f} {unfrozen_final_acc:<20.4f} {unfrozen_final_acc - frozen_final_acc:<20.4f}")
    print(f"{'Trainable Parameters':<30} {trainable_params:<20,} {trainable_params_all:<20,} {trainable_params_all - trainable_params:<20,}")
    print(f"{'Speedup Factor':<30} {'-':<20} {'-':<20} {unfrozen_time/frozen_time:<20.2f}x")
    print("=" * 80)
    
    # Summary
    print("\nSUMMARY:")
    print(f"- Frozen fine-tuning is {unfrozen_time/frozen_time:.2f}x FASTER than full fine-tuning")
    if unfrozen_final_acc > frozen_final_acc:
        print(f"- Full fine-tuning achieves {unfrozen_final_acc - frozen_final_acc:.4f} higher accuracy")
    else:
        print(f"- Frozen fine-tuning achieves {frozen_final_acc - unfrozen_final_acc:.4f} higher accuracy")
    print(f"- Frozen fine-tuning uses only {trainable_params/trainable_params_all*100:.2f}% of trainable parameters")


if __name__ == "__main__":
    main()
