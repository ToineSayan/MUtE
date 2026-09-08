import torch.nn as nn
import torch
from collections import OrderedDict


class MLP(nn.Module):
    """
    Dynamic Resnet network
    """
    def __init__(
            self, 
            input_size, 
            output_size = None, 
            hidden_size = None,
            num_layers = 2, 
            activation = None, # tanh or relu or leakyrelu
            layer_norm = True,
            resnet = False,
            skip_last_linear = False,
            **kwargs
        ):
        super().__init__()

        if output_size is None: output_size = input_size
        if hidden_size is None: hidden_size = input_size

        num_layers = num_layers
        if activation == 'relu' or activation ==  None: act = nn.ReLU()
        if activation == 'leakyrelu': act = nn.LeakyReLU()
        if activation == 'tanh': act = nn.Tanh() 
 

        layers = []
        for l in range(num_layers-1):
            layers += [(f"linear{l+1}", nn.Linear(input_size if l==0 else hidden_size, hidden_size))]
            if layer_norm: layers.append((f"layernorm{l+1}", nn.LayerNorm(hidden_size))) 
            layers.append((f"{activation}{l+1}", act))
        if not skip_last_linear:
            layers += [(f"linear{num_layers}", nn.Linear(hidden_size if num_layers != 1 else input_size, output_size))]

        self.activation = activation
        self.is_resnet = resnet
        self.layers = nn.Sequential(OrderedDict(layers))
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            for layer in self.layers[:-1]: 
                if isinstance(layer, (nn.Linear)):
                    if self.activation == 'tanh': 
                        nn.init.xavier_uniform_(layer.weight)

    def forward(self, x, **kwargs):
        return self.layers(x) if not self.is_resnet else x + self.layers(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def train_model(self, x_train, y_train, x_val, y_val, learning_rate=0.001, num_epochs=100, patience=10, batch_size=512, device=None):
        """
        Train the MLP model with early stopping and best model restoration using mini-batches.
        
        Args:
            x_train: Training input data
            y_train: Training target data
            x_val: Validation input data
            y_val: Validation target data
            learning_rate: Learning rate for optimizer (default: 0.001)
            num_epochs: Number of epochs to train (default: 100)
            patience: Number of epochs with no improvement before early stopping (default: 10)
            batch_size: Batch size for training (default: 512)
            device: Device to train on (default: cuda if available, else cpu)
        
        Returns:
            train_losses: List of training losses per epoch
            val_losses: List of validation losses per epoch
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        best_epoch = 0
        best_model_state = None
        train_losses = []
        val_losses = []
        
        n_samples = x_train.shape[0]
        num_batches = n_samples // batch_size  # Skip incomplete last batch
        
        for epoch in range(num_epochs):
            # Training step with batches
            epoch_loss = 0
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                x_batch = x_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]
                
                optimizer.zero_grad()
                y_pred = self(x_batch)
                train_loss = criterion(y_pred, y_batch)
                train_loss.backward()
                optimizer.step()
                
                epoch_loss += train_loss.item()
            
            # Average loss for the epoch
            avg_epoch_loss = epoch_loss / num_batches
            train_losses.append(avg_epoch_loss)
            
            # Validation step (every 10 epochs or at the last epoch)
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                with torch.no_grad():
                    y_val_pred = self(x_val)
                    val_loss = criterion(y_val_pred, y_val)
                val_losses.append(val_loss.item())
                
                # Check if validation loss improved
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_model_state = {k: v.clone() for k, v in self.state_dict().items()}
                    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_epoch_loss:.4f}, Val Loss: {val_loss.item():.4f} (NEW BEST)")
                else:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_epoch_loss:.4f}, Val Loss: {val_loss.item():.4f}")
                
                # Early stopping check
                if epoch - best_epoch >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}. Best validation loss: {best_val_loss.item():.4f}")
                    break
        
        # Restore the best model parameters
        if best_model_state is not None:
            self.load_state_dict(best_model_state)
            print(f"Model restored to epoch {best_epoch + 1} with validation loss: {best_val_loss.item():.4f}")
        
        return train_losses, val_losses


if __name__ == "__main__":
    
    # Setup model and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = MLP(input_size=10, output_size=10, hidden_size=20, num_layers=3, activation='tanh', layer_norm=False, resnet=True)
    model.to(device)
    print(model)
    print(f"Number of parameters: {model.count_parameters()}")
    
    # Generate training data
    n, d = 100, 10
    x = torch.randn(n, d).to(device)
    y = x.clone().to(device)

    # y = torch.randn(n, d).to(device)

    # Extract validation set
    val_ratio = 0.2
    val_size = int(n * val_ratio)
    x_val = x[:val_size]
    y_val = y[:val_size]
    x_train = x[val_size:]
    y_train = y[val_size:]

    # Train the model
    train_losses, val_losses = model.train_model(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        learning_rate=0.001,
        num_epochs=1000,
        patience=50,
        device=device
    )
    
    print(f"\nTraining complete!")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    if val_losses:
        print(f"Final validation loss: {val_losses[-1]:.4f}")
