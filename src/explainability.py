import os
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def shap_analysis(results, X_train, X_test, feature_names):
    """
    Reproduce SHAP beeswarm + feature importance plots
    (Figures 11, 12, 13 from the paper)

    For each model generate:
    (a) SHAP beeswarm/summary plot
    (b) SHAP feature importance bar plot
    """
    for name, res in results.items():
        model = res['model']
        print(f"\n{'='*50}")
        print(f"SHAP Analysis for {name}")
        print(f"{'='*50}")

        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # (a) Beeswarm / Summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test,
                          feature_names=feature_names,
                          show=False)
        plt.title(f'SHAP Summary Plot — {name}')
        plt.tight_layout()
        plt.savefig(os.path.join(PROJECT_ROOT, f'results/figures/shap_summary_{name}.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()

        # (b) Feature importance bar plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test,
                          feature_names=feature_names,
                          plot_type='bar', show=False)
        plt.title(f'SHAP Feature Importance — {name}')
        plt.tight_layout()
        plt.savefig(os.path.join(PROJECT_ROOT, f'results/figures/shap_importance_{name}.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()


def ice_analysis(results, X_test, feature_names):
    """
    Reproduce Centered ICE (C-ICE) plots with PDP overlay
    (Figures 14, 15, 16 from the paper)

    The paper uses C-ICE plots which show differences relative to
    a fixed point, plus a PDP line showing the average trend.
    """
    # Features to plot ICE for (all input features)
    features_to_plot = list(range(len(feature_names)))

    for name, res in results.items():
        model = res['model']
        print(f"\n{'='*50}")
        print(f"ICE Plot Analysis for {name}")
        print(f"{'='*50}")

        fig, axes = plt.subplots(3, 4, figsize=(24, 18))
        axes = axes.flatten()

        for i, feat_idx in enumerate(features_to_plot):
            if i >= len(axes):
                break
            PartialDependenceDisplay.from_estimator(
                model, X_test, [feat_idx],
                kind='both',  # Both ICE lines and PDP
                centered=True,  # C-ICE (centered)
                ax=axes[i],
                ice_lines_kw={'color': 'blue', 'alpha': 0.1, 'linewidth': 0.5},
                pd_line_kw={'color': 'red', 'linewidth': 2}
            )
            axes[i].set_title(feature_names[feat_idx])

        # Hide unused subplots
        for j in range(len(features_to_plot), len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(f'C-ICE Plots with PDP — {name}', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(PROJECT_ROOT, f'results/figures/ice_plots_{name}.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()
