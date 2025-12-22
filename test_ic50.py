import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import os
import time
from datetime import datetime
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Create a generator with fixed seed
g = torch.Generator()
g.manual_seed(2023)

# Set random seeds for reproducibility
torch.manual_seed(2023)
np.random.seed(2023)
import random
random.seed(2023)

# CUDA settings
torch.cuda.manual_seed(2023)
torch.cuda.manual_seed_all(2023)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Device management
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define data loading function
def load_data(train_path, test_path):
    """Load training and testing data and move to appropriate device"""
    try:
        # Load training data
        train_data = pd.read_csv(train_path, header=0)
        train_features = train_data.iloc[:, 2:-1].values
        train_labels = train_data.iloc[:, -1].values

        # Load testing data
        test_data = pd.read_csv(test_path, header=0)
        test_features = test_data.iloc[:, 2:-1].values
        test_labels = test_data.iloc[:, -1].values

        # Data standardization
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.transform(test_features)

        # Convert to PyTorch tensors and move to device (explicitly specify device)
        X_train = torch.FloatTensor(train_features).to(device)
        y_train = torch.FloatTensor(train_labels).to(device)
        X_test = torch.FloatTensor(test_features).to(device)
        y_test = torch.FloatTensor(test_labels).to(device)

        # Add channel dimension for CNN
        X_train = X_train.unsqueeze(1)
        X_test = X_test.unsqueeze(1)

        return X_train, y_train, X_test, y_test, scaler

    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Define CBAM attention module (improved version)
class AttentionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(AttentionModule, self).__init__()
        # Check if number of channels is reasonable
        if in_channels < reduction_ratio:
            reduction_ratio = max(in_channels // 2, 1)
        reduced_channels = in_channels // reduction_ratio

        # Channel Attention
        self.channel_att = nn.Sequential(
            # Parallel average pooling and max pooling
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(reduced_channels, in_channels, kernel_size=1, bias=False)
        )
        
        self.channel_att_max = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Conv1d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(reduced_channels, in_channels, kernel_size=1, bias=False)
        )
        
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial Attention
        self.spatial_att = nn.Sequential(
            # Input is concatenation of max and average pooling results along channel dimension
            nn.Conv1d(2, 1, kernel_size=7, padding=3, bias=False),  # Use 7x1 convolution kernel
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention calculation
        avg_out = self.channel_att(x)
        max_out = self.channel_att_max(x)
        channel_weight = self.sigmoid_channel(avg_out + max_out)  # Activation after element-wise addition
        x = x * channel_weight  # Apply channel attention

        # Spatial attention calculation
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # Average pooling along channel dimension
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # Max pooling along channel dimension
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)  # Concatenate into 2 channels
        spatial_weight = self.spatial_att(spatial_input)  # Calculate spatial weights
        x = x * spatial_weight  # Apply spatial attention

        return x


# Define residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()

        # Residual path
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)  # Add Dropout layer
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)  # Add Dropout
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Residual connection
        out = self.relu(out)
        return out

# Define CNN model with residual connections
class ResidualCNN(nn.Module):
    def __init__(self, input_channels=1, use_attention=True, initial_channels=16, num_blocks_per_layer=[2, 2, 2],
                 reduction_ratio=8, dropout_rate=0.2, n_features=1, fc_units=32):
        super(ResidualCNN, self).__init__()
        self.use_attention = use_attention
        self.dropout_rate = dropout_rate  # Save incoming dropout_rate

        # Initial convolution layer
        self.conv1 = nn.Conv1d(input_channels, initial_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(initial_channels)
        self.relu = nn.ReLU(inplace=True)

        # Residual blocks
        in_channels = initial_channels
        self.layers = nn.ModuleList()
        for i, num_blocks in enumerate(num_blocks_per_layer):
            out_channels = initial_channels * (2 ** i)
            self.layers.append(self._make_layer(in_channels, out_channels, num_blocks=num_blocks, stride=2 if i > 0 else 1,
                                                dropout_rate=dropout_rate))
            in_channels = out_channels
            if use_attention:
                setattr(self, f'attn{i + 1}', AttentionModule(out_channels, reduction_ratio=reduction_ratio))

        # Pooling layer
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # Calculate fully connected layer input size
        self.n_features = self._calculate_fc_input_size(n_features)

        # Initialize fully connected layer
        self._init_fc_layer(fc_units)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, dropout_rate):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(in_channels, out_channels, stride, dropout_rate))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if self.use_attention:
                attn = getattr(self, f'attn{i + 1}')
                out = attn(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out.squeeze(-1)  # Remove last dimension

    def _calculate_fc_input_size(self, n_features):
        # Calculate feature size after convolution layers and residual blocks
        size = n_features
        # After layer1 (stride=1)
        # After layer2 (stride=2)
        size = (size - 1) // 2 + 1
        # After layer3 (stride=2)
        size = (size - 1) // 2 + 1
        # After global average pooling
        size = 1

        in_channels = self.layers[-1][-1].conv2.out_channels
        return in_channels * size

    def _init_fc_layer(self, fc_units):
        self.fc = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.n_features, fc_units),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(fc_units, 1)  # Output dimension is 1 for regression task
        ).to(next(self.parameters()).device)

# Model evaluation function
def evaluate_model(model, X, y):
    """Evaluate model performance, return R², MSE and RMSE"""
    model.eval()
    with torch.no_grad():
        y_pred = model(X).cpu().numpy()
        y_true = y.cpu().numpy()

    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {
        'r2': r2,
        'mse': mse,
        'rmse': rmse
    }

# Function to train and evaluate model
def train_and_evaluate_model(X_train, y_train, X_test, y_test, params, scaler):
    """Train and evaluate model with given parameters, return performance metrics for training and testing sets"""
    # Extract parameters
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']
    epochs = 40  # Fixed to 30
    patience = 10  # Fixed to 10
    lr_patience = params.get('lr_patience', 5)  # Learning rate scheduler patience
    lr_factor = params.get('lr_factor', 0.5)  # Learning rate reduction factor
    lr_min = params['lr_min']  # Minimum learning rate (changed to discrete)
    initial_channels = params['initial_channels']
    num_blocks_per_layer = [params[f'num_blocks_layer{i + 1}'] for i in range(3)]
    reduction_ratio = params['reduction_ratio']
    dropout_rate = params['dropout_rate']  # Dropout rate (changed to discrete)
    fc_units = params.get('fc_units', 32)  # Number of fully connected layer units

    # Create model and move to device (explicitly specify device)
    model = ResidualCNN(
        input_channels=1,
        use_attention=True,
        initial_channels=initial_channels,
        num_blocks_per_layer=num_blocks_per_layer,
        reduction_ratio=reduction_ratio,
        dropout_rate=dropout_rate,
        n_features=X_train.size(2),
        fc_units=fc_units
    ).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss().to(device)  # Mean squared error loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define learning rate scheduler (based on training set MSE)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_factor, patience=lr_patience,
        min_lr=lr_min, verbose=True
    )

    # Create data loader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)

    # Early stopping mechanism
    best_train_mse = float('inf')  # Smaller MSE is better
    early_stop_counter = 0
    best_model_state = None

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            # Ensure inputs and targets are on correct device (explicitly move)
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward propagation
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward propagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Evaluate on training set
        train_metrics = evaluate_model(model, X_train, y_train)
        train_mse = train_metrics['mse']
        train_r2 = train_metrics['r2']

        # Evaluate on testing set
        test_metrics = evaluate_model(model, X_test, y_test)
        test_mse = test_metrics['mse']
        test_r2 = test_metrics['r2']
        test_rmse = test_metrics['rmse']

        # Update learning rate (based on training set MSE)
        scheduler.step(train_mse)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Early stopping check
        if train_mse < best_train_mse:
            best_train_mse = train_mse
            early_stop_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Epoch {epoch+1}: Early stopping triggered, training set MSE did not improve for {patience} consecutive epochs")
                break

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {running_loss / len(train_loader):.4f}, '
                  f'Train R²: {train_r2:.4f}, Train MSE: {train_mse:.4f}, '
                  f'Test R²: {test_r2:.4f}, Test MSE: {test_mse:.4f}, Test RMSE: {test_rmse:.4f}, LR: {current_lr:.8f}')

    # Load best model and evaluate
    if best_model_state:
        model.load_state_dict(best_model_state)

    final_train_metrics = evaluate_model(model, X_train, y_train)
    final_test_metrics = evaluate_model(model, X_test, y_test)

    print(f"Training set metrics: R²={final_train_metrics['r2']:.4f}, MSE={final_train_metrics['mse']:.4f}, RMSE={final_train_metrics['rmse']:.4f}")
    print(f"Testing set metrics: R²={final_test_metrics['r2']:.4f}, MSE={final_test_metrics['mse']:.4f}, RMSE={final_test_metrics['rmse']:.4f}")
    
    # Release GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    torch.save(model.state_dict(), 'trained_model.pth')
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)


    return final_train_metrics, final_test_metrics

# Main function
def main():
    # Best hyperparameters
    best_params = {
        'learning_rate': 0.01,
        'batch_size': 128,
        'lr_patience': 3,
        'lr_factor': 0.1,
        'lr_min': 1e-06,
        'initial_channels': 128,
        'num_blocks_layer1': 2,
        'num_blocks_layer2': 2,
        'num_blocks_layer3': 4,
        'reduction_ratio': 4,
        'dropout_rate': 0.2,
        'fc_units': 256
    }

    # Replace with your training and testing set file paths
    train_path = '/data/cht/new_ic50/new_affinity_raw_data/train.csv'
    test_path = '/data/cht/new_ic50/new_affinity_raw_data/test.csv'

    # Load data
    data = load_data(train_path, test_path)
    if data is None:
        print("Data loading failed, please check file paths and format.")
        return
    X_train, y_train, X_test, y_test, scaler = data

    # Train and evaluate model
    train_and_evaluate_model(X_train, y_train, X_test, y_test, best_params, scaler)

if __name__ == "__main__":
    main()
