import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate using the 4 metrics from the paper (Equations 4-7):
    - R² (R-squared)
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - MAPE (Mean Absolute Percentage Error)
    """
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred) * 100
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    metrics = {
        'Model': model_name,
        'R2 (%)': round(r2, 3),
        'MAE': round(mae, 3),
        'RMSE': round(rmse, 3),
        'MAPE (%)': round(mape, 3)
    }
    return metrics, y_pred


def evaluate_all_models(results, X_test, y_test):
    """Evaluate all 3 models and produce comparison table (Table 9)"""
    all_metrics = []
    predictions = {}
    for name, res in results.items():
        m, y_pred = evaluate_model(res['model'], X_test, y_test, name)
        all_metrics.append(m)
        predictions[name] = y_pred

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(PROJECT_ROOT, 'results/tables/model_performance.csv'), index=False)
    print("\n=== Model Performance on Test Data (Table 9) ===")
    print(metrics_df.to_string(index=False))
    return metrics_df, predictions


def compute_improvement_table(results, X_test, y_test):
    """Reproduce Table 8"""
    rows = []
    for name, res in results.items():
        test_r2 = res['model'].score(X_test, y_test) * 100
        improvement = test_r2 - res['cv_r2']
        rows.append({
            'Model': name,
            'Train R2 (%)': round(res['train_r2'], 3),
            'CV R2 (%)': round(res['cv_r2'], 3),
            'Test R2 (%)': round(test_r2, 3),
            '% Improvement': round(improvement, 3)
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(PROJECT_ROOT, 'results/tables/improvement_table.csv'), index=False)
    print(df.to_string(index=False))
    return df
