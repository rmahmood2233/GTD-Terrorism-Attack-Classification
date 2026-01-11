"""Visualization helpers for model and experiment outputs."""
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_confusion_matrix(cm, class_names, model_name: str = "Model", save_path: Optional[str] = None):
    plt.figure(figsize=(12, 10))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion'})
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_per_class_metrics(y_test, y_pred, class_names: List[str], model_name: str = "Model", save_path: Optional[str] = None):
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, labels=class_names, zero_division=0)

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, precision, width, label='Precision')
    ax.bar(x, recall, width, label='Recall')
    ax.bar(x + width, f1, width, label='F1-score')

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylabel('Score')
    ax.set_title(f'Per-Class Metrics - {model_name}', fontsize=16, fontweight='bold', pad=20)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_feature_importance(model, feature_names: List[str], model_name: str = "Random Forest", top_n: int = 15, save_path: Optional[str] = None):
    if not hasattr(model, 'feature_importances_'):
        print(f"{model_name} does not have feature_importances_ attribute")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importances[indices][::-1], color='teal')
    plt.yticks(range(top_n), [feature_names[i] for i in indices][::-1])
    plt.title(f'Top {top_n} Feature Importance - {model_name}', fontsize=16)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_model_comparison(results_list, metric: str = 'test_accuracy', save_path: Optional[str] = None):
    model_names = [r['model_name'] for r in results_list]
    scores = [r.get(metric, np.nan) for r in results_list]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=scores, y=model_names, palette='viridis')
    plt.xlabel(metric.replace('_', ' ').title())
    plt.ylabel('Model')
    plt.title(f'Model Comparison - {metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
    plt.xlim(0, 1)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_learning_curves(model, X_train, y_train, model_name: str = "Model", save_path: Optional[str] = None):
    from sklearn.model_selection import learning_curve
    print(f"\nComputing learning curves for {model_name}...")
    train_sizes, train_scores, val_scores = learning_curve(model, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring='accuracy', n_jobs=-1)

    train_scores_mean = train_scores.mean(axis=1)
    val_scores_mean = val_scores.mean(axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='darkorange', label='Training score')
    plt.plot(train_sizes, val_scores_mean, 'o-', color='navy', label='Validation score')
    plt.title(f'Learning Curves - {model_name}')
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()