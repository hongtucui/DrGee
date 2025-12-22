from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import time


def train_and_evaluate(params, X_train_scaled, y_train, X_test_scaled, y_test):
    n_estimators, max_depth = params

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=2023)
    model.fit(X_train_scaled, y_train)

    # 分类预测：获取类别预测和概率预测
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    y_train_proba = model.predict_proba(X_train_scaled)[:, 1]  # 正类概率
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

    # 计算评估指标
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_precision = precision_score(y_train, y_train_pred)
    test_precision = precision_score(y_test, y_test_pred)
    train_recall = recall_score(y_train, y_train_pred)
    test_recall = recall_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    train_auc = roc_auc_score(y_train, y_train_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)

    # 新增：返回预测结果（标签+概率）和指标
    return (train_acc, test_acc, train_precision, test_precision,
            train_recall, test_recall, train_f1, test_f1,
            train_auc, test_auc,
            y_train, y_train_pred, y_train_proba,  # 训练集真实标签、预测标签、预测概率
            y_test, y_test_pred, y_test_proba)     # 测试集真实标签、预测标签、预测概率


def main():
    start = time.time()
    best_params = (300, 10)
    train_file = './train_data.csv'
    test_file = './test_data.csv'

    try:
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)

        X_train = train_df.iloc[:, 2:-1].values.astype(np.float32)
        y_train = train_df.iloc[:, -1].values.astype(np.int32)
        X_test = test_df.iloc[:, 2:-1].values.astype(np.float32)
        y_test = test_df.iloc[:, -1].values.astype(np.int32)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 获取所有返回结果
        metrics = train_and_evaluate(
            best_params, X_train_scaled, y_train, X_test_scaled, y_test
        )
        (train_acc, test_acc, train_precision, test_precision,
         train_recall, test_recall, train_f1, test_f1,
         train_auc, test_auc,
         y_train_true, y_train_pred, y_train_proba,
         y_test_true, y_test_pred, y_test_proba) = metrics

        # 打印评估指标
        print(f"最优参数: n_estimators={best_params[0]}, max_depth={best_params[1]}")
        print(f"训练集: 准确率={train_acc:.4f}, 精确率={train_precision:.4f}, 召回率={train_recall:.4f}, F1={train_f1:.4f}, AUC={train_auc:.4f}")
        print(f"测试集: 准确率={test_acc:.4f}, 精确率={test_precision:.4f}, 召回率={test_recall:.4f}, F1={test_f1:.4f}, AUC={test_auc:.4f}")
        print("\n" + "="*50 + "\n")

        # 打印预测结果（以测试集为例，训练集类似）
        print("测试集预测结果（真实标签 | 预测标签 | 正类概率）：")
        # 组合成DataFrame方便查看
        test_results = pd.DataFrame({
            '真实标签': y_test_true,
            '预测标签': y_test_pred,
            '正类概率': y_test_proba.round(4)  # 保留4位小数
        })

        # 可选：保存结果到CSV
        test_results.to_csv('./test_predictions.csv', index=False)
        print("\n预测结果已保存至 test_predictions.csv")

    except Exception as e:
        print(f"警告: 训练或评估失败: {e}")

    end = time.time()
    print(f"\n执行时间: {end - start:.2f} 秒")


if __name__ == '__main__':
    main()