import time
import tracemalloc

from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV


def train_all_models(X_train, y_train):
    """
    Train XGBoost, GBM, and RF models with GridSearchCV
    as described in Section 4.1 / Table 7 of the paper.
    """
    results = {}

    # ========== 1. RANDOM FOREST ==========
    rf_params = {
        'n_estimators': list(range(60, 221, 40)),  # [60, 100, 140, 180, 220]
        'max_depth': [7],
        'min_samples_split': [3],
        'max_features': ['sqrt']
    }
    rf = RandomForestRegressor(random_state=42)
    rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='r2',
                           n_jobs=-1, verbose=1)

    tracemalloc.start()
    start = time.time()
    rf_grid.fit(X_train, y_train)
    rf_time = time.time() - start
    rf_mem = tracemalloc.get_traced_memory()[1] / (1024 * 1024)  # MB
    tracemalloc.stop()

    results['RF'] = {
        'model': rf_grid.best_estimator_,
        'best_params': rf_grid.best_params_,
        'train_r2': rf_grid.best_estimator_.score(X_train, y_train) * 100,
        'cv_r2': rf_grid.best_score_ * 100,
        'time_secs': rf_time,
        'memory_mb': rf_mem
    }

    # ========== 2. XGBOOST ==========
    xgb_params = {
        'max_depth': list(range(2, 10, 1)),
        'n_estimators': list(range(60, 221, 40)),
        'learning_rate': [0.1, 0.01, 0.05],
        'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
        'gamma': [0]
    }
    xgb = XGBRegressor(random_state=42, objective='reg:squarederror')
    xgb_grid = GridSearchCV(xgb, xgb_params, cv=5, scoring='r2',
                            n_jobs=-1, verbose=1)

    tracemalloc.start()
    start = time.time()
    xgb_grid.fit(X_train, y_train)
    xgb_time = time.time() - start
    xgb_mem = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    tracemalloc.stop()

    results['XGBoost'] = {
        'model': xgb_grid.best_estimator_,
        'best_params': xgb_grid.best_params_,
        'train_r2': xgb_grid.best_estimator_.score(X_train, y_train) * 100,
        'cv_r2': xgb_grid.best_score_ * 100,
        'time_secs': xgb_time,
        'memory_mb': xgb_mem
    }

    # ========== 3. GBM (Gradient Boosting Machine) ==========
    gbm_params = {
        'n_estimators': [10, 15, 19, 20, 21, 50, 100],
        'learning_rate': [0.1, 0.19, 0.2, 0.21, 0.8, 1.0]
    }
    gbm = GradientBoostingRegressor(random_state=42)
    gbm_grid = GridSearchCV(gbm, gbm_params, cv=5, scoring='r2',
                            n_jobs=-1, verbose=1)

    tracemalloc.start()
    start = time.time()
    gbm_grid.fit(X_train, y_train)
    gbm_time = time.time() - start
    gbm_mem = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    tracemalloc.stop()

    results['GBM'] = {
        'model': gbm_grid.best_estimator_,
        'best_params': gbm_grid.best_params_,
        'train_r2': gbm_grid.best_estimator_.score(X_train, y_train) * 100,
        'cv_r2': gbm_grid.best_score_ * 100,
        'time_secs': gbm_time,
        'memory_mb': gbm_mem
    }

    return results
