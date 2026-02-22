import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def plot_correlation_heatmap(df):
    """Pearson Correlation Heatmap"""
    # Encode categoricals for correlation
    df_corr = df.copy()
    df_corr['sex'] = df_corr['sex'].map({'male': 1, 'female': 0})
    df_corr['smoker'] = df_corr['smoker'].map({'yes': 1, 'no': 0})
    df_corr = pd.get_dummies(df_corr, columns=['region'], drop_first=True, dtype=int)

    plt.figure(figsize=(12, 10))
    corr = df_corr.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0,
                fmt='.2f', square=True, linewidths=0.5)
    plt.title('Pearson Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, 'results/figures/correlation_heatmap.png'), dpi=300)
    plt.show()


def plot_feature_vs_target(df):
    """Scatter/box plots of each feature vs charges"""
    features = [col for col in df.columns if col != 'charges']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        if df[feat].dtype == 'object' or df[feat].nunique() <= 6:
            sns.boxplot(x=feat, y='charges', data=df, ax=axes[i])
        else:
            axes[i].scatter(df[feat], df['charges'], alpha=0.5)
            axes[i].set_xlabel(feat)
            axes[i].set_ylabel('Charges ($)')
        axes[i].set_title(f'{feat} vs Charges')

    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, 'results/figures/features_vs_charges.png'), dpi=300)
    plt.show()


def statistical_summary(df):
    """Statistical summary table"""
    summary = df.describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    summary.columns = ['Mean', 'STD', 'Min', 'Q1 (25%)', 'Median (50%)', 'Q3 (75%)', 'Max']
    summary.to_csv(os.path.join(PROJECT_ROOT, 'results/tables/statistical_summary.csv'))
    print(summary.round(2))
    return summary


def plot_learning_curves(results, X_train, y_train):
    """Learning curves for all models"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (name, res) in enumerate(results.items()):
        train_sizes, train_scores, val_scores = learning_curve(
            res['model'], X_train, y_train,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5, scoring='r2', n_jobs=-1
        )

        train_mean = train_scores.mean(axis=1)
        val_mean = val_scores.mean(axis=1)

        axes[idx].plot(train_sizes, train_mean, label='Training score')
        axes[idx].plot(train_sizes, val_mean, label='Validation score')
        axes[idx].set_title(f'{name} Learning Curve')
        axes[idx].set_xlabel('Number of Training Instances')
        axes[idx].set_ylabel('R² Score')
        axes[idx].legend()
        axes[idx].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, 'results/figures/learning_curves.png'), dpi=300)
    plt.show()


def plot_residuals(predictions, y_test, results):
    """Residual plots with Q-Q for all models"""
    fig, axes = plt.subplots(len(predictions), 2, figsize=(14, 5 * len(predictions)))

    for idx, (name, y_pred) in enumerate(predictions.items()):
        residuals = y_test.values - y_pred

        axes[idx, 0].scatter(y_pred, residuals, alpha=0.5)
        axes[idx, 0].axhline(y=0, color='r', linestyle='--')
        axes[idx, 0].set_xlabel('Predicted Values')
        axes[idx, 0].set_ylabel('Residuals')
        axes[idx, 0].set_title(f'{name} Residual Plot')

        stats.probplot(residuals, dist="norm", plot=axes[idx, 1])
        axes[idx, 1].set_title(f'{name} Q-Q Plot')

    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, 'results/figures/residual_plots.png'), dpi=300)
    plt.show()


def plot_prediction_error(predictions, y_test):
    """Actual vs predicted with 45-degree line"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (name, y_pred) in enumerate(predictions.items()):
        axes[idx].scatter(y_test, y_pred, alpha=0.5)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        axes[idx].plot([min_val, max_val], [min_val, max_val],
                       'r--', label='Perfect prediction')
        axes[idx].set_xlabel('Actual Charges ($)')
        axes[idx].set_ylabel('Predicted Charges ($)')
        axes[idx].set_title(f'{name} Prediction Error')
        axes[idx].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, 'results/figures/prediction_error.png'), dpi=300)
    plt.show()


# Need pandas for correlation heatmap encoding
import pandas as pd
