"""src package for GTD pipeline
Export commonly used functions and modules here for convenience.
"""

from .data import load_gtd_data, initial_data_profile
from .pipeline import (
    select_features_objective1,
    handle_missing_values,
    encode_categorical_features,
    engineer_features,
    split_data,
    balance_classes,
    scale_features,
    run_preprocessing_pipeline,
    generate_experiment_combinations,
)
from .models import (
    train_random_forest,
    train_logistic_regression,
    train_gradient_boosting,
    evaluate_model,
)
from .viz import (
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_feature_importance,
    plot_model_comparison,
    plot_learning_curves,
)

__all__ = [
    'load_gtd_data', 'initial_data_profile',
    'select_features_objective1', 'handle_missing_values', 'encode_categorical_features',
    'engineer_features', 'split_data', 'balance_classes', 'scale_features',
    'run_preprocessing_pipeline', 'generate_experiment_combinations',
    'train_random_forest', 'train_logistic_regression', 'train_gradient_boosting', 'evaluate_model',
    'plot_confusion_matrix', 'plot_per_class_metrics', 'plot_feature_importance',
    'plot_model_comparison', 'plot_learning_curves'
]