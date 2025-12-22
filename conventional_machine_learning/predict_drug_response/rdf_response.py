from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import time


def train_and_evaluate(params, X_train_scaled, y_train, X_val_scaled, y_val):
    n_estimators, max_depth = params

    # Change to classifier
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                                  random_state=2023, class_weight='balanced')
    model.fit(X_train_scaled, y_train)

    # Calculate prediction probability (for AUC calculation) and prediction category
    y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
    y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)

    # Calculate binary classification indicators
    train_auc = roc_auc_score(y_train, y_train_proba)
    val_auc = roc_auc_score(y_val, y_val_proba)
    
    # Extra calculation of other auxiliary indicators
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    train_precision = precision_score(y_train, y_train_pred)
    val_precision = precision_score(y_val, y_val_pred)
    train_recall = recall_score(y_train, y_train_pred)
    val_recall = recall_score(y_val, y_val_pred)

    return (n_estimators, max_depth, train_auc, val_auc,
            train_acc, val_acc, train_precision, val_precision,
            train_recall, val_recall)


def main():
    start = time.time()

    fold_files = [
        (f'./fold_data/train_fold_{i}.csv', 
         f'./fold_data/val_fold_{i}.csv') 
         for i in range(1, 6)
    ]
    
    # Define parameters
    n_estimators_list = [50, 100, 200, 300]
    max_depth_list = [10, 20, 30, 40, 50]
    params_list = [(n_estimators, max_depth) for n_estimators in n_estimators_list 
                  for max_depth in max_depth_list]

    best_auc_avg = -np.inf 
    best_params = None
    # Save the best metrics
    best_train_auc_avg = 0
    best_val_auc_avg = 0
    best_train_acc_avg = 0
    best_val_acc_avg = 0
    best_train_precision_avg = 0
    best_val_precision_avg = 0
    best_train_recall_avg = 0
    best_val_recall_avg = 0

    for params in params_list:
        train_auc_vals = []
        val_auc_vals = []
        train_acc_vals = []
        val_acc_vals = []
        train_precision_vals = []
        val_precision_vals = []
        train_recall_vals = []
        val_recall_vals = []

        for train_file, val_file in fold_files:
            try:
                train_df = pd.read_csv(train_file)
                val_df = pd.read_csv(val_file)

                # Extract features and labels
                X_train = train_df.iloc[:, 2:-1].values.astype(np.float32)
                y_train = train_df.iloc[:, -1].values.astype(np.int32) 
                X_val = val_df.iloc[:, 2:-1].values.astype(np.float32)
                y_val = val_df.iloc[:, -1].values.astype(np.int32)

                #Standardized features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)

                # Train and obtain metrics
                results = train_and_evaluate(params, X_train_scaled, y_train, X_val_scaled, y_val)
                (n_estimators, max_depth, train_auc, val_auc,
                 train_acc, val_acc, train_precision, val_precision,
                 train_recall, val_recall) = results

                # Collect various discount indicators
                train_auc_vals.append(train_auc)
                val_auc_vals.append(val_auc)
                train_acc_vals.append(train_acc)
                val_acc_vals.append(val_acc)
                train_precision_vals.append(train_precision)
                val_precision_vals.append(val_precision)
                train_recall_vals.append(train_recall)
                val_recall_vals.append(val_recall)

            except Exception as e:
                print(f"Warning: Parameter {params} failed to train on {train_file} and {val_file}: {e}")

        # Calculate the average indicator
        if train_auc_vals and val_auc_vals:
            train_auc_avg = np.mean(train_auc_vals)
            val_auc_avg = np.mean(val_auc_vals)
            train_acc_avg = np.mean(train_acc_vals)
            val_acc_avg = np.mean(val_acc_vals)
            train_precision_avg = np.mean(train_precision_vals)
            val_precision_avg = np.mean(val_precision_vals)
            train_recall_avg = np.mean(train_recall_vals)
            val_recall_avg = np.mean(val_recall_vals)

            print(f"parameter combination: n_estimators={params[0]}, max_depth={params[1]}:")
            print(f"  Training set: AUC={train_auc_avg:.4f}, accuracy={train_acc_avg:.4f}, "
                  f"precision={train_precision_avg:.4f}, recall={train_recall_avg:.4f}")
            print(f"  validation set: AUC={val_auc_avg:.4f}, accuracy={val_acc_avg:.4f}, "
                  f"precision={val_precision_avg:.4f}, recall={val_recall_avg:.4f}")

            # Update optimal parameters
            if val_auc_avg > best_auc_avg:
                best_auc_avg = val_auc_avg
                best_params = params
                best_train_auc_avg = train_auc_avg
                best_val_auc_avg = val_auc_avg
                best_train_acc_avg = train_acc_avg
                best_val_acc_avg = val_acc_avg
                best_train_precision_avg = train_precision_avg
                best_val_precision_avg = val_precision_avg
                best_train_recall_avg = train_recall_avg
                best_val_recall_avg = val_recall_avg
        else:
            print(f"Error: Parameter {params} failed on all folds")

    if best_params:
        n_estimators_best, max_depth_best = best_params
        print(f"\nFinding the optimal parameters: n_estimators={n_estimators_best}, max_depth={max_depth_best}")
        print(f"Five fold cross validation results:")
        print(f"  Training set: AUC={best_train_auc_avg:.4f}, accuracy={best_train_acc_avg:.4f}, "
              f"precision={best_train_precision_avg:.4f}, recall={best_train_recall_avg:.4f}")
        print(f"  validation set: AUC={best_val_auc_avg:.4f}, accuracy={best_val_acc_avg:.4f}, "
              f"precision={best_val_precision_avg:.4f}, recall={best_val_recall_avg:.4f}")
    else:
        print("Error: Unable to find optimal parameters, please check data and parameter settings")

    end = time.time()
    print(f"Execution time: {end - start:.2f} seconds")


if __name__ == '__main__':
    main()
