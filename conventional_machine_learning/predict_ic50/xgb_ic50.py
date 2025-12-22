from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import pandas as pd
import numpy as np
import time


def train_and_evaluate(params, X_train_scaled, y_train, X_val_scaled, y_val):
    n_estimators, max_depth, lr = params

    model = xgb.XGBRegressor(n_estimators=n_estimators,
                             max_depth=max_depth,
                             learning_rate=lr,
                             objective='reg:squarederror', random_state = 2023)
    model.fit(X_train_scaled, y_train)

   # Calculate the predicted values of the training and validation sets
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)

   # Calculate MSE, R ², and RMSE
    train_mse = mean_squared_error(y_train, y_train_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)

    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)

    train_rmse = np.sqrt(train_mse)
    val_rmse = np.sqrt(val_mse)

    return n_estimators, max_depth, lr, val_mse, train_mse, val_mse, train_r2, val_r2, train_rmse, val_rmse


def main():
    start = time.time()

    fold_files = [
        (f'/data/cht/IC50/new_affinity_raw_data/fold5_raw_data/train_fold_{i}.csv', f'/data/cht/IC50/new_affinity_raw_data/fold5_raw_data/val_fold_{i}.csv') for i in range(1, 6)
    ]
    
    #Define parameters
    n_estimators_list = [50, 100, 200, 300, 500]
    max_depth_list = [3, 4, 5, 6, 7, 8]
    lr_list = [0.01, 0.1, 0.2, 0.3]
    params_list = [(n, m, l) for n in n_estimators_list for m in max_depth_list for l in lr_list]

    best_mse_avg = np.inf  
    best_params = None
    best_train_mse_avg = 0
    best_val_mse_avg = 0
    best_train_r2_avg = 0
    best_val_r2_avg = 0
    best_train_rmse_avg = 0
    best_val_rmse_avg = 0

    for params in params_list:
        train_mse_vals = []
        val_mse_vals = []
        train_r2_vals = []
        val_r2_vals = []
        train_rmse_vals = []
        val_rmse_vals = []

        for train_file, val_file in fold_files:
            try:
                train_df = pd.read_csv(train_file)
                val_df = pd.read_csv(val_file)

                # extract features
                X_train = train_df.iloc[:, 2:-1].values.astype(np.float32)
                y_train = train_df.iloc[:, -1].values.astype(np.float32)
                X_val = val_df.iloc[:, 2:-1].values.astype(np.float32)
                y_val = val_df.iloc[:, -1].values.astype(np.float32)

               # Standardized features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)

                # Train and obtain metrics
                n_estimators, max_depth, lr, val_mse, train_mse, val_mse, train_r2, val_r2, train_rmse, val_rmse = train_and_evaluate(
                    params, X_train_scaled, y_train, X_val_scaled, y_val
                )

                train_mse_vals.append(train_mse)
                val_mse_vals.append(val_mse)
                train_r2_vals.append(train_r2)
                val_r2_vals.append(val_r2)
                train_rmse_vals.append(train_rmse)
                val_rmse_vals.append(val_rmse)

            except Exception as e:
                print(f"Warning: Parameter {params} failed to train on {train_file} and {val_file}: {e}")

        #  Calculate the average indicator
        if train_mse_vals and val_mse_vals:
            train_mse_avg = np.mean(train_mse_vals)
            val_mse_avg = np.mean(val_mse_vals)
            train_r2_avg = np.mean(train_r2_vals)
            val_r2_avg = np.mean(val_r2_vals)
            train_rmse_avg = np.mean(train_rmse_vals)
            val_rmse_avg = np.mean(val_rmse_vals)

            print(f"parameter combination: n_estimators={params[0]}, max_depth={params[1]}, learning_rate={params[2]}:")
            print(f"  Training set: MSE={train_mse_avg:.4f}, R²={train_r2_avg:.4f}, RMSE={train_rmse_avg:.4f}")
            print(f"  Validation set: MSE={val_mse_avg:.4f}, R²={val_r2_avg:.4f}, RMSE={val_rmse_avg:.4f}")

            # Update optimal parameters
            if val_mse_avg < best_mse_avg:
                best_mse_avg = val_mse_avg
                best_params = params
                best_train_mse_avg = train_mse_avg
                best_val_mse_avg = val_mse_avg
                best_train_r2_avg = train_r2_avg
                best_val_r2_avg = val_r2_avg
                best_train_rmse_avg = train_rmse_avg
                best_val_rmse_avg = val_rmse_avg
        else:
            print(f"Error: Parameter {params} failed on all folds")

    if best_params:
        n_estimators_best, max_depth_best, lr_best = best_params
        print(f"\n最优参数找到:Finding the optimal parameters: n_estimators={n_estimators_best}, max_depth={max_depth_best}, learning_rate={lr_best}")
        print(f"Five fold cross validation result:")
        print(f"  Training set: MSE={best_train_mse_avg:.4f}, R²={best_train_r2_avg:.4f}, RMSE={best_train_rmse_avg:.4f}")
        print(f"  Validation set: MSE={best_val_mse_avg:.4f}, R²={best_val_r2_avg:.4f}, RMSE={best_val_rmse_avg:.4f}")
    else:
        print("Error: Unable to find optimal parameters, please check data and parameter settings")

    end = time.time()
    print(f"Execution time: {end - start:.2f} seconds")


if __name__ == '__main__':
    main()
