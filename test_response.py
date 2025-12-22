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
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Create a fixed seed generator
g = torch.Generator()
g.manual_seed(2023)

# Set random seeds
torch.manual_seed(2023)
np.random.seed(2023)
import random
random.seed(2023)

# CUDA settings
torch.cuda.manual_seed(2023)
torch.cuda.manual_seed_all(2023)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define data loading function
def load_data(train_path, test_path):
    
    try:
        # Load training data
        train_data = pd.read_csv(train_path, header=0)
        train_features = train_data.iloc[:, 2:-1].values
        train_labels = train_data.iloc[:, -1].values.astype(np.float32)  

        # Load test data
        test_data = pd.read_csv(test_path, header=0)
        test_features = test_data.iloc[:, 2:-1].values
        test_labels = test_data.iloc[:, -1].values.astype(np.float32)

        # data standardization
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.transform(test_features)

        # Convert to PyTorch tensor and move to device
        X_train = torch.FloatTensor(train_features).to(device)
        y_train = torch.FloatTensor(train_labels).to(device)
        X_test = torch.FloatTensor(test_features).to(device)
        y_test = torch.FloatTensor(test_labels).to(device)

        # Add channel dimension to CNN
        X_train = X_train.unsqueeze(1)
        X_test = X_test.unsqueeze(1)

        return X_train, y_train, X_test, y_test, scaler, train_data.iloc[:, :2], test_data.iloc[:, :2] 

    except Exception as e:
        print(f"Error loading data: {e}")
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
                 reduction_ratio=8, dropout_rate=0.1, n_features=1, fc_units=32):
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
        size = (size - 1) // 2 + 1  # layer2 stride=2
        size = (size - 1) // 2 + 1  # layer3 stride=2
        size = 1  # Global average pooling
        in_channels = self.layers[-1][-1].conv2.out_channels
        return in_channels * size

    def _init_fc_layer(self, fc_units):
        self.fc = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.n_features, fc_units),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(fc_units, 1) 
        ).to(next(self.parameters()).device)

# Model evaluation function
def evaluate_model(model, X, y):
    
    model.eval()
    with torch.no_grad():
        logits = model(X)
        y_pred_prob = torch.sigmoid(logits).cpu().numpy()  
        y_pred = (y_pred_prob > 0.5).astype(int)  
        y_true = y.cpu().numpy()

    # Calculate evaluation indicators
    auc = roc_auc_score(y_true, y_pred_prob)
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
    }, y_true, y_pred_prob

# Functions for training and evaluating models
def train_and_evaluate_model(X_train, y_train, X_test, y_test, train_ids, test_ids, params, scaler):
   
  
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']
    epochs = 100
    patience = 10
    lr_patience = params.get('lr_patience', 5)
    lr_factor = params.get('lr_factor', 0.5)
    lr_min = params['lr_min']
    initial_channels = params['initial_channels']
    num_blocks_per_layer = [params[f'num_blocks_layer{i + 1}'] for i in range(3)]
    reduction_ratio = params['reduction_ratio']
    dropout_rate = params['dropout_rate']
    fc_units = params.get('fc_units', 32)

    # Create a model
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

    # Define loss function
    criterion = nn.BCEWithLogitsLoss().to(device)  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_factor, patience=lr_patience,
        min_lr=lr_min, verbose=True
    )

    # Create data loader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)

    # Early stop mechanism
    best_train_loss = float('inf')
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

        
        train_metrics, _, _ = evaluate_model(model, X_train, y_train)
        test_metrics, _, _ = evaluate_model(model, X_test, y_test)
        train_loss = running_loss / len(train_loader)

        # Update learning rate
        scheduler.step(train_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Early stop inspection
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            early_stop_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Epoch {epoch+1}: Early stop triggered, training loss has not improved for consecutive {patience} rounds")
                break

        # Printing progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, '
                  f'Train AUC: {train_metrics["auc"]:.4f}, Train Acc: {train_metrics["accuracy"]:.4f}, '
                  f'Test AUC: {test_metrics["auc"]:.4f}, Test Acc: {test_metrics["accuracy"]:.4f}, LR: {current_lr:.8f}')

    # Load the best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Final evaluation
    final_train_metrics, train_true, train_pred_prob = evaluate_model(model, X_train, y_train)
    final_test_metrics, test_true, test_pred_prob = evaluate_model(model, X_test, y_test)

    print(f"Training set metrics: AUC={final_train_metrics['auc']:.4f}, Acc={final_train_metrics['accuracy']:.4f}, "
          f"Precision={final_train_metrics['precision']:.4f}, Recall={final_train_metrics['recall']:.4f}, "
          f"F1={final_train_metrics['f1']:.4f}")
    print(f"Testing set metrics: AUC={final_test_metrics['auc']:.4f}, Acc={final_test_metrics['accuracy']:.4f}, "
          f"Precision={final_test_metrics['precision']:.4f}, Recall={final_test_metrics['recall']:.4f}, "
          f"F1={final_test_metrics['f1']:.4f}")

    # Save prediction results
    train_results = pd.concat([
        train_ids.reset_index(drop=True),
        pd.DataFrame({'true_label': train_true, 'pred_prob': train_pred_prob, 
                      'pred_label': (train_pred_prob > 0.5).astype(int)})
    ], axis=1)
    test_results = pd.concat([
        test_ids.reset_index(drop=True),
        pd.DataFrame({'true_label': test_true, 'pred_prob': test_pred_prob, 
                      'pred_label': (test_pred_prob > 0.5).astype(int)})
    ], axis=1)

    train_results.to_csv('train_predictions.csv', index=False)
    test_results.to_csv('test_predictions.csv', index=False)
    print("The predicted results have been saved to train_predictions.csv and test_predictions.csv")

    # Save model and scaler
    torch.save(model.state_dict(), 'trained_binary_model.pth')
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # release cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_train_metrics, final_test_metrics


def main():
    
    best_params = {
        'learning_rate': 0.0001,
        'batch_size': 128,
        'lr_patience': 5,
        'lr_factor': 0.3,
        'lr_min': 1e-07,
        'initial_channels': 32,
        'num_blocks_layer1': 2,
        'num_blocks_layer2': 3,
        'num_blocks_layer3': 4,
        'reduction_ratio': 8,
        'dropout_rate': 0.1,
        'fc_units': 64
    }

    # data path
    train_path = './train_data.csv'
    test_path = './test_data.csv'

    # Load data
    data = load_data(train_path, test_path)
    if data is None:
        print("Data loading failed, please check the file path and format.")
        return
    X_train, y_train, X_test, y_test, scaler, train_ids, test_ids = data

    # Training and evaluation
    train_and_evaluate_model(X_train, y_train, X_test, y_test, train_ids, test_ids, best_params, scaler)

if __name__ == "__main__":
    main()