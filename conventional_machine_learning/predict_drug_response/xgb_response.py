from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import pandas as pd
import numpy as np
import time


def train_and_evaluate(params, X_train_scaled, y_train, X_val_scaled, y_val):
    n_estimators, max_depth, lr = params

    
    model = xgb.XGBClassifier(n_estimators=n_estimators,
                             max_depth=max_depth,
                             learning_rate=lr,
                             objective='binary:logistic',  
                             eval_metric='logloss', 
                             random_state=2023)
    model.fit(X_train_scaled, y_train)

    #Calculate the predicted values (probability and category) for the training and validation sets
    y_train_pred_proba = model.predict_proba(X_train_scaled)[:, 1]  
    y_val_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
    
    y_train_pred = model.predict(X_train_scaled)  
    y_val_pred = model.predict(X_val_scaled)

    #Calculate binary evaluation indicators
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    
    train_precision = precision_score(y_train, y_train_pred)
    val_precision = precision_score(y_val, y_val_pred)
    
    train_recall = recall_score(y_train, y_train_pred)
    val_recall = recall_score(y_val, y_val_pred)
    
    train_f1 = f1_score(y_train, y_train_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    
    train_auc = roc_auc_score(y_train, y_train_pred_proba)
    val_auc = roc_auc_score(y_val, y_val_pred_proba)

    return (n_estimators, max_depth, lr, 
            train_acc, val_acc, 
            train_precision, val_precision, 
            train_recall, val_recall, 
            train_f1, val_f1, 
            train_auc, val_auc)


def main():
    start = time.time()

    fold_files = [
        (f'./fold_data/train_fold_{i}.csv', 
         f'./fold_data/val_fold_{i}.csv') for i in range(1, 6)
    ]

    n_estimators_list = [50, 100, 200, 300, 500]
    max_depth_list = [3, 4, 5, 6, 7, 8]
    lr_list = [0.01, 0.1, 0.2, 0.3]
    params_list = [(n, m, l) for n in n_estimators_list for m in max_depth_list for l in lr_list]

    best_auc_avg = -np.inf  
    best_params = None
    
    #Evaluation indicators corresponding to the optimal parameters
    best_metrics = {
        'train_acc': 0, 'val_acc': 0,
        'train_precision': 0, 'val_precision': 0,
        'train_recall': 0, 'val_recall': 0,
        'train_f1': 0, 'val_f1': 0,
        'train_auc': 0, 'val_auc': 0
    }

    for params in params_list:
        metrics = {
            'train_acc': [], 'val_acc': [],
            'train_precision': [], 'val_precision': [],
            'train_recall': [], 'val_recall': [],
            'train_f1': [], 'val_f1': [],
            'train_auc': [], 'val_auc': []
        }

        for train_file, val_file in fold_files:
            try:
                train_df = pd.read_csv(train_file)
                val_df = pd.read_csv(val_file)

                # Extract features and labels
                X_train = train_df.iloc[:, 2:-1].values.astype(np.float32)
                y_train = train_df.iloc[:, -1].values.astype(np.int32)  
                X_val = val_df.iloc[:, 2:-1].values.astype(np.float32)
                y_val = val_df.iloc[:, -1].values.astype(np.int32)

                # Standardized features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)

                # Train and obtain metrics
                results = train_and_evaluate(params, X_train_scaled, y_train, X_val_scaled, y_val)
                (n_estimators, max_depth, lr, 
                 train_acc, val_acc, 
                 train_precision, val_precision, 
                 train_recall, val_recall, 
                 train_f1, val_f1, 
                 train_auc, val_auc) = results

                # Save the indicators of each fold
                metrics['train_acc'].append(train_acc)
                metrics['val_acc'].append(val_acc)
                metrics['train_precision'].append(train_precision)
                metrics['val_precision'].append(val_precision)
                metrics['train_recall'].append(train_recall)
                metrics['val_recall'].append(val_recall)
                metrics['train_f1'].append(train_f1)
                metrics['val_f1'].append(val_f1)
                metrics['train_auc'].append(train_auc)
                metrics['val_auc'].append(val_auc)

            except Exception as e:
                print(f"Warning: Parameter {params} failed to train on {train_file} and {val_file}: {e}")

        # Calculate the average indicator
        if metrics['val_auc']:  
            # Calculate the average value of each indicator
            avg_metrics = {k: np.mean(v) for k, v in metrics.items()}

            print(f"parameter combination n_estimators={params[0]}, max_depth={params[1]}, learning_rate={params[2]}:")
            print(f"  Training set: ACC={avg_metrics['train_acc']:.4f}, Precision={avg_metrics['train_precision']:.4f}, "
                  f"Recall={avg_metrics['train_recall']:.4f}, F1={avg_metrics['train_f1']:.4f}, AUC={avg_metrics['train_auc']:.4f}")
            print(f"  Validation set: ACC={avg_metrics['val_acc']:.4f}, Precision={avg_metrics['val_precision']:.4f}, "
                  f"Recall={avg_metrics['val_recall']:.4f}, F1={avg_metrics['val_f1']:.4f}, AUC={avg_metrics['val_auc']:.4f}")

            # Update optimal parameters
            if avg_metrics['val_auc'] > best_auc_avg:
                best_auc_avg = avg_metrics['val_auc']
                best_params = params
                best_metrics = avg_metrics
        else:
            print(f"Error: Parameter {params} failed on all folds")

    if best_params:
        n_estimators_best, max_depth_best, lr_best = best_params
        print(f"\nFinding the optimal parameters: n_estimators={n_estimators_best}, max_depth={max_depth_best}, learning_rate={lr_best}")
        print(f"Five fold cross validation results:")
        print(f"  Training set: ACC={best_metrics['train_acc']:.4f}, Precision={best_metrics['train_precision']:.4f}, "
              f"Recall={best_metrics['train_recall']:.4f}, F1={best_metrics['train_f1']:.4f}, AUC={best_metrics['train_auc']:.4f}")
        print(f"  Validation set: ACC={best_metrics['val_acc']:.4f}, Precision={best_metrics['val_precision']:.4f}, "
              f"Recall={best_metrics['val_recall']:.4f}, F1={best_metrics['val_f1']:.4f}, AUC={best_metrics['val_auc']:.4f}")
    else:
        print("Error: Unable to find optimal parameters, please check data and parameter settings")

    end = time.time()
    print(f"Execution time: {end - start:.2f} seconds")


if __name__ == '__main__':
    main()