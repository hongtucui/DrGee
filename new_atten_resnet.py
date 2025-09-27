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

# 创建一个固定种子的生成器
g = torch.Generator()
g.manual_seed(2023)

# 设置随机种子确保结果可复现
torch.manual_seed(2023)
np.random.seed(2023)
import random
random.seed(2023)

# CUDA设置
torch.cuda.manual_seed(2023)
torch.cuda.manual_seed_all(2023)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# 设备管理
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义数据加载函数
def load_data(fold_id, data_dir="/data/cht/new_ic50/new_affinity_raw_data/fold5_raw_data"):
    """加载指定折的训练和验证数据并移至正确设备"""
    try:
        # 加载训练数据
        train_data = pd.read_csv(os.path.join(data_dir, f"train_fold_{fold_id}.csv"), header=0)
        train_features = train_data.iloc[:, 2:-1].values
        train_labels = train_data.iloc[:, -1].values

        # 加载验证数据
        val_data = pd.read_csv(os.path.join(data_dir, f"val_fold_{fold_id}.csv"), header=0)
        val_features = val_data.iloc[:, 2:-1].values
        val_labels = val_data.iloc[:, -1].values

        # 数据标准化
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        val_features = scaler.transform(val_features)

        # 转换为PyTorch张量并移至设备（显式指定设备）
        X_train = torch.FloatTensor(train_features).to(device)
        y_train = torch.FloatTensor(train_labels).to(device)
        X_val = torch.FloatTensor(val_features).to(device)
        y_val = torch.FloatTensor(val_labels).to(device)

        # 为CNN添加通道维度
        X_train = X_train.unsqueeze(1)
        X_val = X_val.unsqueeze(1)

        return X_train, y_train, X_val, y_val

    except Exception as e:
        print(f"加载第{fold_id}折数据时出错: {e}")
        return None

# 定义CBAM注意力模块（改进版）
class AttentionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(AttentionModule, self).__init__()
        # 检查通道数是否合理
        if in_channels < reduction_ratio:
            reduction_ratio = max(in_channels // 2, 1)
        reduced_channels = in_channels // reduction_ratio

        # 通道注意力 (Channel Attention)
        self.channel_att = nn.Sequential(
            # 并行的平均池化和最大池化
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

        # 空间注意力 (Spatial Attention)
        self.spatial_att = nn.Sequential(
            # 输入为通道维度的最大和平均池化结果拼接
            nn.Conv1d(2, 1, kernel_size=7, padding=3, bias=False),  # 使用7x7卷积核
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力计算
        avg_out = self.channel_att(x)
        max_out = self.channel_att_max(x)
        channel_weight = self.sigmoid_channel(avg_out + max_out)  # 元素相加后激活
        x = x * channel_weight  # 通道注意力作用

        # 空间注意力计算
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # 沿通道维度平均池化
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # 沿通道维度最大池化
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)  # 拼接成2通道
        spatial_weight = self.spatial_att(spatial_input)  # 计算空间权重
        x = x * spatial_weight  # 空间注意力作用

        return x


# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()

        # 残差路径
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)  # 添加Dropout层
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 捷径连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)  # 添加Dropout
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 残差连接
        out = self.relu(out)
        return out

# 定义带残差连接的CNN模型
class ResidualCNN(nn.Module):
    def __init__(self, input_channels=1, use_attention=True, initial_channels=16, num_blocks_per_layer=[2, 2, 2],
                 reduction_ratio=8, dropout_rate=0.2, n_features=1, fc_units=32):
        super(ResidualCNN, self).__init__()
        self.use_attention = use_attention
        self.dropout_rate = dropout_rate  # 保存传入的dropout_rate

        # 初始卷积层
        self.conv1 = nn.Conv1d(input_channels, initial_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(initial_channels)
        self.relu = nn.ReLU(inplace=True)

        # 残差块
        in_channels = initial_channels
        self.layers = nn.ModuleList()
        for i, num_blocks in enumerate(num_blocks_per_layer):
            out_channels = initial_channels * (2 ** i)
            self.layers.append(self._make_layer(in_channels, out_channels, num_blocks=num_blocks, stride=2 if i > 0 else 1,
                                                dropout_rate=dropout_rate))
            in_channels = out_channels
            if use_attention:
                setattr(self, f'attn{i + 1}', AttentionModule(out_channels, reduction_ratio=reduction_ratio))

        # 池化层
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # 计算全连接层输入大小
        self.n_features = self._calculate_fc_input_size(n_features)

        # 初始化全连接层
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
        return out.squeeze(-1)  # 去除最后一个维度

    def _calculate_fc_input_size(self, n_features):
        # 计算经过卷积层和残差块后的特征大小
        size = n_features
        # 初始卷积层和ReLU
        # 经过layer1 (stride=1)
        # 经过layer2 (stride=2)
        size = (size - 1) // 2 + 1
        # 经过layer3 (stride=2)
        size = (size - 1) // 2 + 1
        # 经过全局平均池化
        size = 1

        in_channels = self.layers[-1][-1].conv2.out_channels
        return in_channels * size

    def _init_fc_layer(self, fc_units):
        self.fc = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.n_features, fc_units),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(fc_units, 1)  # 回归任务输出维度为1
        ).to(next(self.parameters()).device)

# 模型评估函数
def evaluate_model(model, X, y):
    """评估模型性能，返回R²、MSE和RMSE"""
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

# 训练和评估模型的函数
def train_and_evaluate_model(X_train, y_train, X_val, y_val, params):
    """使用给定参数训练并评估模型，返回验证集性能指标"""
    # 提取参数
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']
    epochs = 30  # 固定为30
    patience = 10  # 固定为10
    lr_patience = params.get('lr_patience', 5)  # 学习率调度耐心值
    lr_factor = params.get('lr_factor', 0.5)  # 学习率降低因子
    lr_min = params['lr_min']  # 最小学习率（已改为离散）
    initial_channels = params['initial_channels']
    num_blocks_per_layer = [params[f'num_blocks_layer{i + 1}'] for i in range(3)]
    reduction_ratio = params['reduction_ratio']
    dropout_rate = params['dropout_rate']  # Dropout率（已改为离散）
    fc_units = params.get('fc_units', 32)  # 全连接层单元数

    # 创建模型并移至设备（显式指定设备）
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

    # 定义损失函数和优化器
    criterion = nn.MSELoss().to(device)  # 均方误差损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 定义学习率调度器（基于验证MSE）
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_factor, patience=lr_patience,
        min_lr=lr_min, verbose=True
    )

    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)

    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 早停机制
    best_val_mse = float('inf')  # MSE越小越好
    early_stop_counter = 0
    best_model_state = None

    # 训练循环
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            # 确保输入和目标在正确的设备上（显式移动）
            inputs = inputs.to(device)
            targets = targets.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 验证
        val_metrics = evaluate_model(model, X_val, y_val)
        val_mse = val_metrics['mse']
        val_r2 = val_metrics['r2']

        # 训练集评估
        train_metrics = evaluate_model(model, X_train, y_train)
        train_mse = train_metrics['mse']
        train_r2 = train_metrics['r2']

        # 更新学习率（基于验证MSE）
        scheduler.step(val_mse)

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 早停检查
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            early_stop_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Epoch {epoch+1}: 早停触发，验证MSE连续{patience}轮未改善")
                break

        # 打印进度
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {running_loss / len(train_loader):.4f}, '
                  f'Train R²: {train_r2:.4f}, Train MSE: {train_mse:.4f}, '
                  f'Val R²: {val_r2:.4f}, Val MSE: {val_mse:.4f}, LR: {current_lr:.8f}')

    # 加载最佳模型并评估
    if best_model_state:
        model.load_state_dict(best_model_state)

    final_train_metrics = evaluate_model(model, X_train, y_train)
    final_val_metrics = evaluate_model(model, X_val, y_val)
    print(f"训练集指标: R²={final_train_metrics['r2']:.4f}, MSE={final_train_metrics['mse']:.4f}, RMSE={final_train_metrics['rmse']:.4f}")
    print(f"验证集指标: R²={final_val_metrics['r2']:.4f}, MSE={final_val_metrics['mse']:.4f}, RMSE={final_val_metrics['rmse']:.4f}")

    # 释放GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_train_metrics, final_val_metrics

# 超参数优化的目标函数
def objective(params, data_folds):
    """Hyperopt的目标函数 - 在所有折上评估参数并返回平均性能"""
    # 在所有折上评估
    all_fold_train_metrics = []
    all_fold_val_metrics = []

    for fold_id, fold_data in enumerate(data_folds):
        if fold_data is None:
            continue
        X_train, y_train, X_val, y_val = fold_data
        print(f"\n=== 在第 {fold_id + 1}/5 折上评估参数 ===")

        # 训练并评估模型
        train_metrics, val_metrics = train_and_evaluate_model(
            X_train, y_train,
            X_val, y_val,
            params
        )

        all_fold_train_metrics.append(train_metrics)
        all_fold_val_metrics.append(val_metrics)

        print(f"  第 {fold_id + 1} 折: Train R²={train_metrics['r2']:.4f}, Train MSE={train_metrics['mse']:.4f}, Train RMSE={train_metrics['rmse']:.4f}, "
              f"Val R²={val_metrics['r2']:.4f}, Val MSE={val_metrics['mse']:.4f}, Val RMSE={val_metrics['rmse']:.4f}")

    if len(all_fold_val_metrics) == 0:
        print("没有可用的折数据进行评估！")
        return {
            'loss': float('inf'),
            'status': STATUS_OK,
            'params': params,
            'train_fold_metrics': [],
            'val_fold_metrics': [],
            'avg_train_metrics': {},
            'avg_val_metrics': {}
        }

    # 计算平均性能
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

    print(f"\n=== 参数评估结果 ===")
    print(f"  平均训练指标: R²={avg_train_metrics['r2']:.4f}, MSE={avg_train_metrics['mse']:.4f}, RMSE={avg_train_metrics['rmse']:.4f}")
    print(f"  平均验证指标: R²={avg_val_metrics['r2']:.4f}, MSE={avg_val_metrics['mse']:.4f}, RMSE={avg_val_metrics['rmse']:.4f}")

    return {
        'loss': avg_val_metrics['mse'],  # 最小化MSE
        'status': STATUS_OK,
        'params': params,
        'train_fold_metrics': all_fold_train_metrics,
        'val_fold_metrics': all_fold_val_metrics,
        'avg_train_metrics': avg_train_metrics,
        'avg_val_metrics': avg_val_metrics
    }

# 保存最优参数到JSON文件
def save_best_params(best_params, best_metrics, output_dir="results"):
    """将最优参数和性能指标保存到JSON文件"""
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 保存参数
        with open(f"{output_dir}/best_hyperparameters.json", 'w') as f:
            json.dump(best_params, f, indent=4)

        # 保存性能指标
        with open(f"{output_dir}/best_performance_metrics.json", 'w') as f:
            json.dump(best_metrics, f, indent=4)

        print(f"\n=== 结果已保存到 {output_dir} ===")
        print(f"最优参数: {best_params}")
        print(f"平均训练性能: R²={best_metrics['avg_train_metrics']['r2']:.4f}, MSE={best_metrics['avg_train_metrics']['mse']:.4f}, RMSE={best_metrics['avg_train_metrics']['rmse']:.4f}")
        print(f"平均验证性能: R²={best_metrics['avg_val_metrics']['r2']:.4f}, MSE={best_metrics['avg_val_metrics']['mse']:.4f}, RMSE={best_metrics['avg_val_metrics']['rmse']:.4f}")
    except Exception as e:
        print(f"保存结果时出错: {e}")

# 主函数
def main():
    # 创建保存结果的目录
    os.makedirs('results', exist_ok=True)

    # 定义超参数搜索空间（将learning_rate、lr_min、dropout_rate改为离散值）
    space = {
        # 学习率（离散对数空间值）
        'learning_rate': hp.choice('learning_rate', [1e-5, 1e-4, 1e-3, 1e-2]),
        'batch_size': hp.choice('batch_size', [128]),
        'lr_patience': hp.choice('lr_patience', [3, 5, 7]),
        'lr_factor': hp.choice('lr_factor', [0.1, 0.3, 0.5]),
        # 最小学习率（离散对数空间值）
        'lr_min': hp.choice('lr_min', [1e-7, 1e-6, 1e-5]),
        'initial_channels': hp.choice('initial_channels', [32, 64, 128]),
        'num_blocks_layer1': hp.choice('num_blocks_layer1', [2, 3, 4]),
        'num_blocks_layer2': hp.choice('num_blocks_layer2', [2, 3, 4]),
        'num_blocks_layer3': hp.choice('num_blocks_layer3', [2, 3, 4]),
        'reduction_ratio': hp.choice('reduction_ratio', [4, 8, 16]),
        # Dropout率（离散常见值）
        'dropout_rate': hp.choice('dropout_rate', [0.1, 0.2, 0.3, 0.4, 0.5]),
        'fc_units': hp.choice('fc_units', [64, 128, 256])
    }

    # 加载所有折的数据
    data_folds = []

    print("\n=== 加载所有折的数据 ===")
    for fold_id in range(1, 6):
        fold_data = load_data(fold_id)
        if fold_data is not None:
            data_folds.append(fold_data)
            print(f"第 {fold_id}/5 折数据加载成功")
        else:
            print(f"跳过第{fold_id}折数据加载")

    if len(data_folds) == 0:
        print("没有可用的折数据！")
        return

    # 超参数优化
    print("\n=== 开始超参数优化（基于五折交叉验证的平均MSE）===")
    trials = Trials()
    best = fmin(
        fn=lambda params: objective(params, data_folds),
        space=space,
        algo=tpe.suggest,
        max_evals=500,  # 可以根据需要调整评估次数
        trials=trials,
        rstate=np.random.default_rng(2023)
    )

    # 提取最佳参数和性能
    best_trial = trials.best_trial
    best_params = best_trial['result']['params']
    best_metrics = best_trial['result']

    print("\n=== 最佳超参数 ===")
    print(f"  学习率: {best_params['learning_rate']}")
    print(f"  批大小: {best_params['batch_size']}")
    print(f"  学习率调度耐心值: {best_params['lr_patience']}")
    print(f"  学习率降低因子: {best_params['lr_factor']}")
    print(f"  最小学习率: {best_params['lr_min']}")
    print(f"  初始卷积层通道数: {best_params['initial_channels']}")
    print(f"  第一层残差块数量: {best_params['num_blocks_layer1']}")
    print(f"  第二层残差块数量: {best_params['num_blocks_layer2']}")
    print(f"  第三层残差块数量: {best_params['num_blocks_layer3']}")
    print(f"  注意力模块减少比例: {best_params['reduction_ratio']}")
    print(f"  Dropout率: {best_params['dropout_rate']:.4f}")
    print(f"  全连接层单元数: {best_params['fc_units']}")

    print("\n=== 五折交叉验证的性能 ===")
    for fold_id, (train_metrics, val_metrics) in enumerate(zip(best_metrics['train_fold_metrics'], best_metrics['val_fold_metrics'])):
        print(f"  第 {fold_id + 1} 折: Train R²={train_metrics['r2']:.4f}, Train MSE={train_metrics['mse']:.4f}, Train RMSE={train_metrics['rmse']:.4f}, "
              f"Val R²={val_metrics['r2']:.4f}, Val MSE={val_metrics['mse']:.4f}, Val RMSE={val_metrics['rmse']:.4f}")

    print("\n=== 平均性能 ===")
    print(f"  平均训练 R²: {best_metrics['avg_train_metrics']['r2']:.4f}")


if __name__ == "__main__":
    main()
