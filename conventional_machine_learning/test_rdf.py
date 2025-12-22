from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import time


def train_and_evaluate(params, X_train_scaled, y_train, X_test_scaled, y_test):
    n_estimators, max_depth = params

    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=2023)
    model.fit(X_train_scaled, y_train)

    # Calculate the predicted values for the training and testing sets
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

   # Calculate MSE, R ², and RMSE
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)

    return train_mse, test_mse, train_r2, test_r2, train_rmse, test_rmse


def main():
    start = time.time()

    # optimal parameters
    best_params = (300, 20)  

    # Read the training and testing sets
    train_file = '/data/cht/IC50/new_affinity_raw_data/train.csv'
    test_file = '/data/cht/IC50/new_affinity_raw_data/test.csv'

    try:
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)

        # extract features
        X_train = train_df.iloc[:, 2:-1].values.astype(np.float32)
        y_train = train_df.iloc[:, -1].values.astype(np.float32)
        X_test = test_df.iloc[:, 2:-1].values.astype(np.float32)
        y_test = test_df.iloc[:, -1].values.astype(np.float32)

       # Standardized features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

         # Train and obtain metrics
        train_mse, test_mse, train_r2, test_r2, train_rmse, test_rmse = train_and_evaluate(
            best_params, X_train_scaled, y_train, X_test_scaled, y_test
        )

        print(f"Optimal parameters: n_estimators={best_params[0]}, max_depth={best_params[1]}")
        print(f"Training set: MSE={train_mse:.4f}, R²={train_r2:.4f}, RMSE={train_rmse:.4f}")
        print(f"Test set: MSE={test_mse:.4f}, R²={test_r2:.4f}, RMSE={test_rmse:.4f}")

    except Exception as e:
        print(f"Warning: Training or evaluation failure: {e}")

    end = time.time()
    print(f"Execution time: {end - start:.2f} seconds")


if __name__ == '__main__':
    main()