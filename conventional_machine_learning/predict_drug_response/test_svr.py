from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import time


def train_and_evaluate(params, X_train_scaled, y_train, X_test_scaled, y_test):
    c, g = params

    model = SVC(kernel='rbf', C=c, gamma=g, probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)

    #Calculate the predicted category and probability
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    y_train_proba = model.predict_proba(X_train_scaled)[:, 1] 
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

    
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

  
    return (train_acc, test_acc, train_precision, test_precision,
            train_recall, test_recall, train_f1, test_f1,
            train_auc, test_auc,
            y_train, y_train_pred, y_train_proba,  
            y_test, y_test_pred, y_test_proba)


def main():
    start = time.time()
    best_params = (1, 0.1)
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

        metrics = train_and_evaluate(
            best_params, X_train_scaled, y_train, X_test_scaled, y_test
        )
        (train_acc, test_acc, train_precision, test_precision,
         train_recall, test_recall, train_f1, test_f1,
         train_auc, test_auc,
         y_train_true, y_train_pred, y_train_proba,
         y_test_true, y_test_pred, y_test_proba) = metrics

        
        print(f"Optimal parameters: C={best_params[0]}, gamma={best_params[1]}")
        print(f"Training set: accuracy={train_acc:.4f}, precision={train_precision:.4f}, "
              f"recall={train_recall:.4f}, F1={train_f1:.4f}, AUC={train_auc:.4f}")
        print(f"test set: accuracy={test_acc:.4f}, precision={test_precision:.4f}, "
              f"recall={test_recall:.4f}, F1={test_f1:.4f}, AUC={test_auc:.4f}")

        
        test_results = pd.DataFrame({
            'true label': y_test_true,
            'predicted label': y_test_pred,
            'Positive prediction probability': y_test_proba
        })
        test_results.to_csv('./test_predictions.csv', index=False)
        print("The predicted results of the test set have been saved to test_predictions.csv")


    except Exception as e:
        print(f"Warning: Training or evaluation failure: {e}")

    end = time.time()
    print(f"Execution time: {end - start:.2f} seconds")


if __name__ == '__main__':
    main()