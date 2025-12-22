import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import os
import time
from datetime import datetime
import json
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

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
def load_data(fold_id, data_dir="/data/cht/new_ic50/new_affinity_raw_data/fold5_raw_data"):
    """Load training and validation data for the specified fold and move to the correct device"""
    try:
        # Load training data
        train_data = pd.read_csv(os.path.join(data_dir, f"train_fold_{fold_id}.csv"), header=0)
        train_features = train_data.iloc[:, 2:-1].values
        train_labels = train_data.iloc[:, -1].values

        # Load validation data
        val_data = pd.read_csv(os.path.join(data_dir, f"val_fold_{fold_id}.csv"), header=0)
        val_features = val_data.iloc[:, 2:-1].values
        val_labels = val_data.iloc[:, -1].values

        # Data standardization
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        val_features = scaler.transform(val_features)

        # Convert to PyTorch tensors and move to device (explicitly specify device)
        X_train = torch.FloatTensor(train_features).to(device)
        y_train = torch.FloatTensor(train_labels).to(device)
        X_val = torch.FloatTensor(val_features).to(device)
        y_val = torch.FloatTensor(val_labels).to(device)

        # Add channel dimension for CNN
        X_train = X_train.unsqueeze(1)
        X_val = X_val.unsqueeze(1)

        return X_train, y_train, X_val, y_val

    except Exception as e:
        print(f"Error loading fold {fold_id} data: {e}")
        return None

# Define CBAM attention module (improved version)
class AttentionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(AttentionModule, self).__init__()
        # Check if the number of channels is reasonable
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
            nn.Conv1d(2, 1, kernel_size=7, padding=3, bias=False),  # Using 7x7 convolution kernel
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
        out = self.dropout(out)  # Apply Dropout
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
        self.dropout_rate = dropout_rate  # Save the input dropout_rate

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

        # Calculate input size for fully connected layer
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
        return out.squeeze(-1)  # Remove the last dimension

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

# Function to train and evaluate the model
def train_and_evaluate_model(X_train, y_train, X_val, y_val, params):
    """Train and evaluate the model with given parameters, return validation performance metrics"""
    # Extract parameters
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']
    epochs = 30  # Fixed at 30
    patience = 10  # Fixed at 10
    lr_patience = params.get('lr_patience', 5)  # Learning rate scheduler patience
    lr_factor = params.get('lr_factor', 0.5)  # Learning rate reduction factor
    lr_min = params['lr_min']  # Minimum learning rate (changed to discrete)
    initial_channels = params['initial_channels']
    num_blocks_per_layer = [params[f'num_blocks_layer{i + 1}'] for i in range(3)]
    reduction_ratio = params['reduction_ratio']
    dropout_rate = params['dropout_rate']  # Dropout rate (changed to discrete)
    fc_units = params.get('fc_units', 32)  # Number of units in fully connected layer

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

    # Define learning rate scheduler (based on validation MSE)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_factor, patience=lr_patience,
        min_lr=lr_min, verbose=True
    )

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)

    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Early stopping mechanism
    best_val_mse = float('inf')  # Lower MSE is better
    early_stop_counter = 0
    best_model_state = None

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            # Ensure inputs and targets are on the correct device (explicitly move)
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation
        val_metrics = evaluate_model(model, X_val, y_val)
        val_mse = val_metrics['mse']
        val_r2 = val_metrics['r2']

        # Training set evaluation
        train_metrics = evaluate_model(model, X_train, y_train)
        train_mse = train_metrics['mse']
        train_r2 = train_metrics['r2']

        # Update learning rate (based on validation MSE)
        scheduler.step(val_mse)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Early stopping check
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            early_stop_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Epoch {epoch+1}: Early stopping triggered, validation MSE did not improve for {patience} consecutive epochs")
                break

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {running_loss / len(train_loader):.4f}, '
                  f'Train R²: {train_r2:.4f}, Train MSE: {train_mse:.4f}, '
                  f'Val R²: {val_r2:.4f}, Val MSE: {val_mse:.4f}, LR: {current_lr:.8f}')

    # Load best model and evaluate
    if best_model_state:
        model.load_state_dict(best_model_state)

    final_train_metrics = evaluate_model(model, X_train, y_train)
    final_val_metrics = evaluate_model(model, X_val, y_val)
    print(f"Training set metrics: R²={final_train_metrics['r2']:.4f}, MSE={final_train_metrics['mse']:.4f}, RMSE={final_train_metrics['rmse']:.4f}")
    print(f"Validation set metrics: R²={final_val_metrics['r2']:.4f}, MSE={final_val_metrics['mse']:.4f}, RMSE={final_val_metrics['rmse']:.4f}")

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_train_metrics, final_val_metrics

# Objective function for hyperparameter optimization
def objective(params, data_folds):
    """Hyperopt objective function - evaluate parameters on all folds and return average performance"""
    # Evaluate on all folds
    all_fold_train_metrics = []
    all_fold_val_metrics = []

    for fold_id, fold_data in enumerate(data_folds):
        if fold_data is None:
            continue
        X_train, y_train, X_val, y_val = fold_data
        print(f"\n=== Evaluating parameters on fold {fold_id + 1}/5 ===")

        # Train and evaluate the model
        train_metrics, val_metrics = train_and_evaluate_model(
            X_train, y_train,
            X_val, y_val,
            params
        )

        all_fold_train_metrics.append(train_metrics)
        all_fold_val_metrics.append(val_metrics)

        print(f"  Fold {fold_id + 1}: Train R²={train_metrics['r2']:.4f}, Train MSE={train_metrics['mse']:.4f}, Train RMSE={train_metrics['rmse']:.4f}, "
              f"Val R²={val_metrics['r2']:.4f}, Val MSE={val_metrics['mse']:.4f}, Val RMSE={val_metrics['rmse']:.4f}")

    if len(all_fold_val_metrics) == 0:
        print("No fold data available for evaluation!")
        return {
            'loss': float('inf'),
            'status': STATUS_OK,
            'params': params,
            'train_fold_metrics': [],
            'val_fold_metrics': [],
            'avg_train_metrics': {},
            'avg_val_metrics': {}
        }

    # Calculate average performance
    avg_train_metrics = {
        'r2': np.mean([m['r2'] for m in all_fold_train_metrics]),
        'mse': np.mean([m['mse'] for m in all_fold_train_metrics]),
        'rmse': np.mean([m['rmse'] for m in all_fold_train_metrics])
    }

    avg_val_metrics = {
        'r2': np.mean([m['r2'] for m in all_fold_val_metrics]),
        'mse': np.mean([m['mse'] for m in all_fold_val_metrics]),
        'rmse': np.mean([m['rmse'] for m in all_fold_val_metrics])
    }

    print(f"\n=== Parameter evaluation results ===")
    print(f"  Average training metrics: R²={avg_train_metrics['r2']:.4f}, MSE={avg_train_metrics['mse']:.4f}, RMSE={avg_train_metrics['rmse']:.4f}")
    print(f"  Average validation metrics: R²={avg_val_metrics['r2']:.4f}, MSE={avg_val_metrics['mse']:.4f}, RMSE={avg_val_metrics['rmse']:.4f}")

    return {
        'loss': avg_val_metrics['mse'],  # Minimize MSE
        'status': STATUS_OK,
        'params': params,
        'train_fold_metrics': all_fold_train_metrics,
        'val_fold_metrics': all_fold_val_metrics,
        'avg_train_metrics': avg_train_metrics,
        'avg_val_metrics': avg_val_metrics
    }

# Save best parameters to JSON file
def save_best_params(best_params, best_metrics, output_dir="results"):
    """Save best parameters and performance metrics to JSON files"""
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Save parameters
        with open(f"{output_dir}/best_hyperparameters.json", 'w') as f:
            json.dump(best_params, f, indent=4)

        # Save performance metrics
        with open(f"{output_dir}/best_performance_metrics.json", 'w') as f:
            json.dump(best_metrics, f, indent=4)

        print(f"\n=== Results saved to {output_dir} ===")
        print(f"Best parameters: {best_params}")
        print(f"Average training performance: R²={best_metrics['avg_train_metrics']['r2']:.4f}, MSE={best_metrics['avg_train_metrics']['mse']:.4f}, RMSE={best_metrics['avg_train_metrics']['rmse']:.4f}")
        print(f"Average validation performance: R²={best_metrics['avg_val_metrics']['r2']:.4f}, MSE={best_metrics['avg_val_metrics']['mse']:.4f}, RMSE={best_metrics['avg_val_metrics']['rmse']:.4f}")
    except Exception as e:
        print(f"Error saving results: {e}")

# Main function
def main():
    # Create directory to save results
    os.makedirs('results', exist_ok=True)

    # Define hyperparameter search space (changed learning_rate, lr_min, dropout_rate to discrete values)
    space = {
        # Learning rate (discrete logarithmic space values)
        'learning_rate': hp.choice('learning_rate', [1e-5, 1e-4, 1e-3, 1e-2]),
        'batch_size': hp.choice('batch_size', [128]),
        'lr_patience': hp.choice('lr_patience', [3, 5, 7]),
        'lr_factor': hp.choice('lr_factor', [0.1, 0.3, 0.5]),
        # Minimum learning rate (discrete logarithmic space values)
        'lr_min': hp.choice('lr_min', [1e-7, 1e-6, 1e-5]),
        'initial_channels': hp.choice('initial_channels', [32, 64, 128]),
        'num_blocks_layer1': hp.choice('num_blocks_layer1', [2, 3, 4]),
        'num_blocks_layer2': hp.choice('num_blocks_layer2', [2, 3, 4]),
        'num_blocks_layer3': hp.choice('num_blocks_layer3', [2, 3, 4]),
        'reduction_ratio': hp.choice('reduction_ratio', [4, 8, 16]),
        # Dropout rate (common discrete values)
        'dropout_rate': hp.choice('dropout_rate', [0.1, 0.2, 0.3, 0.4, 0.5]),
        'fc_units': hp.choice('fc_units', [64, 128, 256])
    }

    # Load data for all folds
    data_folds = []

    print("\n=== Loading data for all folds ===")
    for fold_id in range(1, 6):
        fold_data = load_data(fold_id)
        if fold_data is not None:
            data_folds.append(fold_data)
            print(f"Fold {fold_id}/5 data loaded successfully")
        else:
            print(f"Skipping fold {fold_id} data loading")

    if len(data_folds) == 0:
        print("No fold data available!")
        return

    # Hyperparameter optimization
    print("\n=== Starting hyperparameter optimization (based on average MSE from 5-fold cross-validation) ===")
    trials = Trials()
    best = fmin(
        fn=lambda params: objective(params, data_folds),
        space=space,
        algo=tpe.suggest,
        max_evals=500,  # Can be adjusted as needed
        trials=trials,
        rstate=np.random.default_rng(2023)
    )

    # Extract best parameters and performance
    best_trial = trials.best_trial
    best_params = best_trial['result']['params']
    best_metrics = best_trial['result']

    print("\n=== Best hyperparameters ===")
    print(f"  Learning rate: {best_params['learning_rate']}")
    print(f"  Batch size: {best_params['batch_size']}")
    print(f"  Learning rate scheduler patience: {best_params['lr_patience']}")
    print(f"  Learning rate reduction factor: {best_params['lr_factor']}")
    print(f"  Minimum learning rate: {best_params['lr_min']}")
    print(f"  Initial convolution layer channels: {best_params['initial_channels']}")
    print(f"  Number of residual blocks in layer 1: {best_params['num_blocks_layer1']}")
    print(f"  Number of residual blocks in layer 2: {best_params['num_blocks_layer2']}")
    print(f"  Number of residual blocks in layer 3: {best_params['num_blocks_layer3']}")
    print(f"  Attention module reduction ratio: {best_params['reduction_ratio']}")
    print(f"  Dropout rate: {best_params['dropout_rate']:.4f}")
    print(f"  Number of fully connected layer units: {best_params['fc_units']}")

    print("\n=== Performance from 5-fold cross-validation ===")
    for fold_id, (train_metrics, val_metrics) in enumerate(zip(best_metrics['train_fold_metrics'], best_metrics['val_fold_metrics'])):
        print(f"  Fold {fold_id + 1}: Train R²={train_metrics['r2']:.4f}, Train MSE={train_metrics['mse']:.4f}, Train RMSE={train_metrics['rmse']:.4f}, "
              f"Val R²={val_metrics['r2']:.4f}, Val MSE={val_metrics['mse']:.4f}, Val RMSE={val_metrics['rmse']:.4f}")

    print("\n=== Average performance ===")
    print(f"  Average training R²: {best_metrics['avg_train_metrics']['r2']:.4f}")


if __name__ == "__main__":
    main()
