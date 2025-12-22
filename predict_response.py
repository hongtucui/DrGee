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
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Create a fixed seed generator
g = torch.Generator()
g.manual_seed(2023)

# Set random seeds to ensure reproducible results
torch.manual_seed(2023)
np.random.seed(2023)
import random
random.seed(2023)

# CUDA settings
torch.cuda.manual_seed(2023)
torch.cuda.manual_seed_all(2023)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Equipment Management
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using equipment: {device}")

# Define data loading function
def load_data(fold_id, data_dir="/data/cht/response/fold_data"):
    """Load the training and validation data of the specified fold and move it to the correct device"""
    try:
        # Load training data
        train_data = pd.read_csv(os.path.join(data_dir, f"train_fold_{fold_id}.csv"), header=0)
        train_features = train_data.iloc[:, 2:-1].values
        # The last column is the binary label (0 or 1)
        train_labels = train_data.iloc[:, -1].values.astype(int)

        # Load validation data
        val_data = pd.read_csv(os.path.join(data_dir, f"val_fold_{fold_id}.csv"), header=0)
        val_features = val_data.iloc[:, 2:-1].values
        val_labels = val_data.iloc[:, -1].values.astype(int)

        # data standardization
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        val_features = scaler.transform(val_features)

        # Convert to PyTorch tensor and move to device
        X_train = torch.FloatTensor(train_features).to(device)
        y_train = torch.FloatTensor(train_labels).to(device) 
        X_val = torch.FloatTensor(val_features).to(device)
        y_val = torch.FloatTensor(val_labels).to(device)

        # Add channel dimension to CNN
        X_train = X_train.unsqueeze(1)
        X_val = X_val.unsqueeze(1)

        return X_train, y_train, X_val, y_val

    except Exception as e:
        print(f"Error loading {fold_id} fold data: {e}")
        return None

# Define CBAM Attention Module
class AttentionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(AttentionModule, self).__init__()
        if in_channels < reduction_ratio:
            reduction_ratio = max(in_channels // 2, 1)
        reduced_channels = in_channels // reduction_ratio

        # channel attention
        self.channel_att = nn.Sequential(
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

        # spatial attention
        self.spatial_att = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.channel_att(x)
        max_out = self.channel_att_max(x)
        channel_weight = self.sigmoid_channel(avg_out + max_out)
        x = x * channel_weight

        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_weight = self.spatial_att(spatial_input)
        x = x * spatial_weight

        return x


# Define residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# Define a CNN model with residual connections
class ResidualCNN(nn.Module):
    def __init__(self, input_channels=1, use_attention=True, initial_channels=16, num_blocks_per_layer=[2, 2, 2],
                 reduction_ratio=8, dropout_rate=0.2, n_features=1, fc_units=32):
        super(ResidualCNN, self).__init__()
        self.use_attention = use_attention
        self.dropout_rate = dropout_rate

        self.conv1 = nn.Conv1d(input_channels, initial_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(initial_channels)
        self.relu = nn.ReLU(inplace=True)

        in_channels = initial_channels
        self.layers = nn.ModuleList()
        for i, num_blocks in enumerate(num_blocks_per_layer):
            out_channels = initial_channels * (2 ** i)
            self.layers.append(self._make_layer(in_channels, out_channels, num_blocks=num_blocks, stride=2 if i > 0 else 1,
                                                dropout_rate=dropout_rate))
            in_channels = out_channels
            if use_attention:
                setattr(self, f'attn{i + 1}', AttentionModule(out_channels, reduction_ratio=reduction_ratio))

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.n_features = self._calculate_fc_input_size(n_features)

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
        return out.squeeze(-1)

    def _calculate_fc_input_size(self, n_features):
        size = n_features
        size = (size - 1) // 2 + 1
        size = (size - 1) // 2 + 1
        size = 1

        in_channels = self.layers[-1][-1].conv2.out_channels
        return in_channels * size

    def _init_fc_layer(self, fc_units):
        # Binary output layer
        self.fc = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.n_features, fc_units),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(fc_units, 1)
        ).to(next(self.parameters()).device)

# Model evaluation function
def evaluate_model(model, X, y):
    """Evaluate model performance"""
    model.eval()
    with torch.no_grad():
        logits = model(X)
        y_pred_proba = torch.sigmoid(logits).cpu().numpy()  # Convert to probability
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()  # Convert to Category
        y_true = y.cpu().numpy().astype(int)

    # Calculate AUC
    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        auc = 0.5 
        
    # Calculate other classification indicators
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        'auc': auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Functions for training and evaluating models
def train_and_evaluate_model(X_train, y_train, X_val, y_val, params):
    """Train and evaluate the model using given parameters, and return performance metrics for the validation set"""
    # extract parameters
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']
    epochs = 30
    patience = 10
    lr_patience = params.get('lr_patience', 5)
    lr_factor = params.get('lr_factor', 0.5)
    lr_min = params['lr_min']
    initial_channels = params['initial_channels']
    num_blocks_per_layer = [params[f'num_blocks_layer{i + 1}'] for i in range(3)]
    reduction_ratio = params['reduction_ratio']
    dropout_rate = params['dropout_rate']
    fc_units = params.get('fc_units', 32)

    # Create a model and move it to the device
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

    # Define binary loss function and optimizer
    criterion = nn.BCEWithLogitsLoss().to(device) 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=lr_factor, patience=lr_patience,
        min_lr=lr_min, verbose=True
    )

    # Create data loader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)

    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Early stop mechanism
    best_val_auc = -float('inf')
    early_stop_counter = 0
    best_model_state = None

    # Training cycle
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # forward propagation
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # verify
        val_metrics = evaluate_model(model, X_val, y_val)
        val_auc = val_metrics['auc']
        val_acc = val_metrics['accuracy']

        # Training set evaluation
        train_metrics = evaluate_model(model, X_train, y_train)
        train_auc = train_metrics['auc']
        train_acc = train_metrics['accuracy']

        # Update learning rate
        scheduler.step(val_auc)

        # Obtain the current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Early stop inspection
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            early_stop_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Epoch {epoch+1}: Early stop triggered, verifying that AUC has not improved for consecutive {patience} rounds")
                break

        # Printing progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {running_loss / len(train_loader):.4f}, '
                  f'Train AUC: {train_auc:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.8f}')

    # Load the best model and evaluate it
    if best_model_state:
        model.load_state_dict(best_model_state)

    final_train_metrics = evaluate_model(model, X_train, y_train)
    final_val_metrics = evaluate_model(model, X_val, y_val)
    print(f"Training set metrics: AUC={final_train_metrics['auc']:.4f}, Acc={final_train_metrics['accuracy']:.4f}, "
          f"Precision={final_train_metrics['precision']:.4f}, Recall={final_train_metrics['recall']:.4f}, F1={final_train_metrics['f1']:.4f}")
    print(f"Validation set metrics: AUC={final_val_metrics['auc']:.4f}, Acc={final_val_metrics['accuracy']:.4f}, "
          f"Precision={final_val_metrics['precision']:.4f}, Recall={final_val_metrics['recall']:.4f}, F1={final_val_metrics['f1']:.4f}")

    # Release GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_train_metrics, final_val_metrics

# The objective function of hyperparameter optimization
def objective(params, data_folds):
    """Hyperopt's objective function - evaluate parameters on all folds and return average performance"""
    all_fold_train_metrics = []
    all_fold_val_metrics = []

    for fold_id, fold_data in enumerate(data_folds):
        if fold_data is None:
            continue
        X_train, y_train, X_val, y_val = fold_data
        print(f"\n===Evaluate parameters on the {fold_id+1}/5th fold===")

        train_metrics, val_metrics = train_and_evaluate_model(
            X_train, y_train,
            X_val, y_val,
            params
        )

        all_fold_train_metrics.append(train_metrics)
        all_fold_val_metrics.append(val_metrics)

        print(f"  The {fold_id + 1} fold: Train AUC={train_metrics['auc']:.4f}, Train Acc={train_metrics['accuracy']:.4f}, "
              f"Val AUC={val_metrics['auc']:.4f}, Val Acc={val_metrics['accuracy']:.4f}")

    if len(all_fold_val_metrics) == 0:
        print("No available discount data for evaluation!")
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
        'auc': np.mean([m['auc'] for m in all_fold_train_metrics]),
        'accuracy': np.mean([m['accuracy'] for m in all_fold_train_metrics]),
        'precision': np.mean([m['precision'] for m in all_fold_train_metrics]),
        'recall': np.mean([m['recall'] for m in all_fold_train_metrics]),
        'f1': np.mean([m['f1'] for m in all_fold_train_metrics])
    }

    avg_val_metrics = {
        'auc': np.mean([m['auc'] for m in all_fold_val_metrics]),
        'accuracy': np.mean([m['accuracy'] for m in all_fold_val_metrics]),
        'precision': np.mean([m['precision'] for m in all_fold_val_metrics]),
        'recall': np.mean([m['recall'] for m in all_fold_val_metrics]),
        'f1': np.mean([m['f1'] for m in all_fold_val_metrics])
    }

    print(f"\n=== Parameter evaluation results ===")
    print(f"  Average training metrics: AUC={avg_train_metrics['auc']:.4f}, Acc={avg_train_metrics['accuracy']:.4f}")
    print(f"  Average validation metrics: AUC={avg_val_metrics['auc']:.4f}, Acc={avg_val_metrics['accuracy']:.4f}")

    return {
        'loss': 1 - avg_val_metrics['auc'], 
        'status': STATUS_OK,
        'params': params,
        'train_fold_metrics': all_fold_train_metrics,
        'val_fold_metrics': all_fold_val_metrics,
        'avg_train_metrics': avg_train_metrics,
        'avg_val_metrics': avg_val_metrics
    }

# Save the optimal parameters to a JSON file
def save_best_params(best_params, best_metrics, output_dir="results"):

    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(f"{output_dir}/best_hyperparameters.json", 'w') as f:
            json.dump(best_params, f, indent=4)

        with open(f"{output_dir}/best_performance_metrics.json", 'w') as f:
            json.dump(best_metrics, f, indent=4)

        print(f"\n=== The result has been saved to {output_dir} ===")
        print(f"Optimal parameters: {best_params}")
        print(f"Average training performance: AUC={best_metrics['avg_train_metrics']['auc']:.4f}, Acc={best_metrics['avg_train_metrics']['accuracy']:.4f}")
        print(f"Average validation performance: AUC={best_metrics['avg_val_metrics']['auc']:.4f}, Acc={best_metrics['avg_val_metrics']['accuracy']:.4f}")
    except Exception as e:
        print(f"Error saving results: {e}")


def main():
    os.makedirs('results', exist_ok=True)

    # Define hyperparameter search space
    space = {
        'learning_rate': hp.choice('learning_rate', [1e-5, 1e-4, 1e-3, 1e-2]),
        'batch_size': hp.choice('batch_size', [128]),
        'lr_patience': hp.choice('lr_patience', [3, 5, 7]),
        'lr_factor': hp.choice('lr_factor', [0.1, 0.3, 0.5]),
        'lr_min': hp.choice('lr_min', [1e-7, 1e-6, 1e-5]),
        'initial_channels': hp.choice('initial_channels', [32, 64, 128]),
        'num_blocks_layer1': hp.choice('num_blocks_layer1', [2, 3, 4]),
        'num_blocks_layer2': hp.choice('num_blocks_layer2', [2, 3, 4]),
        'num_blocks_layer3': hp.choice('num_blocks_layer3', [2, 3, 4]),
        'reduction_ratio': hp.choice('reduction_ratio', [4, 8, 16]),
        'dropout_rate': hp.choice('dropout_rate', [0.1, 0.2, 0.3, 0.4, 0.5]),
        'fc_units': hp.choice('fc_units', [64, 128, 256])
    }

    #Load all folded data
    data_folds = []

    print("\n=== Load all folded data ===")
    for fold_id in range(1, 6):
        fold_data = load_data(fold_id)
        if fold_data is not None:
            data_folds.append(fold_data)
            print(f"The {fold_id}/5th fold data loaded successfully")
        else:
            print(f"Skip loading data for the {fold_id} th fold")

    if len(data_folds) == 0:
        print("No available discount data!")
        return

    # hyperparameter optimization
    print("\n=== Start hyperparameter optimization (based on the average AUC of five fold cross validation)===")
    trials = Trials()
    best = fmin(
        fn=lambda params: objective(params, data_folds),
        space=space,
        algo=tpe.suggest,
        max_evals=500,
        trials=trials,
        rstate=np.random.default_rng(2023)
    )

    # Extract optimal parameters and performance
    best_trial = trials.best_trial
    best_params = best_trial['result']['params']
    best_metrics = best_trial['result']

    print("\n=== Best hyperparameters ===")
    print(f"  Learning rate: {best_params['learning_rate']}")
    print(f"  Batch size: {best_params['batch_size']}")
    print(f"  Learning rate scheduling patience value: {best_params['lr_patience']}")
    print(f"  Learning rate reduction factor: {best_params['lr_factor']}")
    print(f"  Minimum learning rate: {best_params['lr_min']}")
    print(f"  Initial number of channels in the convolutional layer: {best_params['initial_channels']}")
    print(f"  Number of residual blocks in the first layer: {best_params['num_blocks_layer1']}")
    print(f"  Number of residual blocks in the second layer: {best_params['num_blocks_layer2']}")
    print(f"  Number of residual blocks in the third layer: {best_params['num_blocks_layer3']}")
    print(f"  Attention module reduction ratio: {best_params['reduction_ratio']}")
    print(f"  Dropout rate: {best_params['dropout_rate']:.4f}")
    print(f"  Number of fully connected layer units: {best_params['fc_units']}")

    print("\n=== Performance of five fold cross validation ===")
    for fold_id, (train_metrics, val_metrics) in enumerate(zip(best_metrics['train_fold_metrics'], best_metrics['val_fold_metrics'])):
        print(f"  The {fold_id+1} fold: Train AUC={train_metrics['auc']:.4f}, Train Acc={train_metrics['accuracy']:.4f}, "
              f"Val AUC={val_metrics['auc']:.4f}, Val Acc={val_metrics['accuracy']:.4f}")

    print("\n=== Average performance ===")
    print(f"  Average training AUC: {best_metrics['avg_train_metrics']['auc']:.4f}")
    print(f"  Average verification AUC: {best_metrics['avg_val_metrics']['auc']:.4f}")

    # save results
    save_best_params(best_params, best_metrics)


if __name__ == "__main__":
    main()
