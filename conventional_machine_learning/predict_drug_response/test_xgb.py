from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import pandas as pd
import numpy as np
import time


def train_and_evaluate(params, X_train_scaled, y_train, X_test_scaled, y_test):
    n_estimators, max_depth, lr = params

    model = xgb.XGBClassifier(n_estimators=n_estimators,
                             max_depth=max_depth,
                             learning_rate=lr,
                             objective='binary:logistic',
                             random_state=2023,
                             use_label_encoder=False,
                             eval_metric='logloss')

    model.fit(X_train_scaled, y_train)

    
    y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    
    train_auc = roc_auc_score(y_train, y_train_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    train_precision = precision_score(y_train, y_train_pred)
    test_precision = precision_score(y_test, y_test_pred)
    
    train_recall = recall_score(y_train, y_train_pred)
    test_recall = recall_score(y_test, y_test_pred)
    
    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    
    return (train_auc, test_auc, train_acc, test_acc,
            train_precision, test_precision, train_recall, test_recall,
            train_f1, test_f1,
            y_test, y_test_pred, y_test_proba) 


def main():
    start = time.time()

    best_params = (500, 8, 0.01)

    train_file = './train_data.csv'
    test_file = './test_data.csv'
    
    result_save_path = './test_prediction_results.csv'

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

       
        (train_auc, test_auc, train_acc, test_acc,
         train_precision, test_precision, train_recall, test_recall,
         train_f1, test_f1,
         true_labels, pred_labels, pred_probs) = metrics

    
        print(f"Optimal parameters: n_estimators={best_params[0]}, max_depth={best_params[1]}, learning_rate={best_params[2]}")
        print(f"Training set: AUC={train_auc:.4f}, accuracy={train_acc:.4f}, precision={train_precision:.4f}, recall={train_recall:.4f}, F1={train_f1:.4f}")
        print(f"test set: AUC={test_auc:.4f}, accuracy={test_acc:.4f}, precision={test_precision:.4f}, recall={test_recall:.4f}, F1={test_f1:.4f}")

        
        result_df = pd.DataFrame({
            'true label': true_labels,
            'predicted label': pred_labels,
            'Positive prediction probability': pred_probs
        })
        result_df.to_csv(result_save_path, index=False)
        

    except Exception as e:
        print(f"Warning: Training or evaluation failure: {e}")

    end = time.time()
    print(f"Execution time: {end - start:.2f} seconds")


if __name__ == '__main__':
    main()