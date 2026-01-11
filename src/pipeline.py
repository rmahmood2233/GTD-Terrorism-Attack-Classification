"""Plug-and-play preprocessing pipeline components.

This module collects reusable functions originally implemented in the notebook
so they can be imported and used programmatically in scripts and tests.
"""
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import defaultdict
from itertools import product


def select_features_objective1(df: pd.DataFrame, target: str = 'attacktype1_txt') -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Select features for attack-type classification (objective 1).

    The function chooses a sensible default set of features if present, and
    returns X, y, and feature names.
    """
    print("\n" + "=" * 60)
    print("BLOCK: FEATURE SELECTION (OBJECTIVE 1)")
    print("=" * 60)

    candidate_features = [
        'iyear', 'imonth', 'iday', 'country_txt', 'region_txt', 'latitude', 'longitude',
        'city', 'nkill', 'nwound', 'gname', 'weaptype1_txt', 'targtype1_txt'
    ]

    available_features = [c for c in candidate_features if c in df.columns]
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe")

    X = df[available_features].copy()
    y = df[target].copy()

    print(f"Features selected: {len(available_features)}")
    print(f"   Target variable: {target}")
    print(f"   Dataset shape: {X.shape}")
    print(f"   Number of classes: {y.nunique()}")

    return X, y, available_features


def handle_missing_values(X: pd.DataFrame, y: pd.Series, strategy: str = 'impute') -> Tuple[pd.DataFrame, pd.Series]:
    """Handle missing values using different strategies: 'drop', 'impute', 'impute_flag'."""
    print("\n" + "=" * 60)
    print("BLOCK: MISSING VALUE HANDLING")
    print("=" * 60)

    X_processed = X.copy()
    y_processed = y.copy()

    if strategy == 'drop':
        print("Applying dropna() to remove rows with any missing value")
        df_concat = pd.concat([X_processed, y_processed], axis=1)
        df_concat = df_concat.dropna()
        X_processed = df_concat[X_processed.columns]
        y_processed = df_concat[y_processed.name]

    elif strategy in ('impute', 'impute_flag'):
        num_cols = X_processed.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X_processed.select_dtypes(include=['object', 'category']).columns.tolist()

        if num_cols:
            num_imputer = SimpleImputer(strategy='median')
            X_processed[num_cols] = num_imputer.fit_transform(X_processed[num_cols])

        if cat_cols:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            X_processed[cat_cols] = cat_imputer.fit_transform(X_processed[cat_cols])

        if strategy == 'impute_flag':
            # add indicator columns for missingness (before imputation)
            for col in X.columns:
                if X[col].isnull().any():
                    X_processed[col + '_missing_flag'] = X[col].isnull().astype(int)

    else:
        raise ValueError(f"Unknown missing value strategy: {strategy}")

    print(f"Final shape: {X_processed.shape}")
    print(f"   Remaining missing: {X_processed.isnull().sum().sum()}")

    return X_processed, y_processed


def encode_categorical_features(X: pd.DataFrame, y: pd.Series = None, strategy: str = 'target', n_jobs: int = 1) -> pd.DataFrame:
    """Encode categorical features using 'onehot', 'target', or 'grouped'.

    If 'target' is chosen, y must be provided.
    """
    print("\n" + "=" * 60)
    print("BLOCK: CATEGORICAL ENCODING")
    print("=" * 60)

    X_encoded = X.copy()
    cat_cols = X_encoded.select_dtypes(include=['object', 'category']).columns.tolist()

    if strategy == 'onehot':
        X_encoded = pd.get_dummies(X_encoded, columns=cat_cols, dummy_na=False)

    elif strategy == 'target':
        if y is None:
            raise ValueError("Target encoding requires target variable 'y'.")
        for col in cat_cols:
            targ_mean = pd.concat([X_encoded[col], y], axis=1).groupby(col)[y.name].mean()
            X_encoded[col + '_te'] = X_encoded[col].map(targ_mean)
            X_encoded.drop(columns=[col], inplace=True)

    elif strategy == 'grouped':
        # Reduce cardinality by grouping rare categories into 'OTHER'
        for col in cat_cols:
            freq = X_encoded[col].value_counts(normalize=True)
            rare = freq[freq < 0.01].index
            X_encoded[col] = X_encoded[col].replace(rare, 'OTHER')
            # label-encode grouped categories
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        print(f"   Applied label encoding to grouped features")

    else:
        raise ValueError(f"Unknown encoding strategy: {strategy}")

    print(f"Encoding complete. Final shape: {X_encoded.shape}")

    return X_encoded


def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """Create a set of commonly useful derived features."""
    print("\n" + "=" * 60)
    print("BLOCK: FEATURE ENGINEERING")
    print("=" * 60)

    X_eng = X.copy()
    initial_features = X_eng.shape[1]

    # Absolute latitude
    if 'latitude' in X_eng.columns:
        X_eng['lat_abs'] = X_eng['latitude'].abs()

    # Quarter from month
    if 'imonth' in X_eng.columns:
        X_eng['quarter'] = ((X_eng['imonth'] - 1) // 3 + 1).fillna(0).astype(int)

    # Total casualties
    if 'nkill' in X_eng.columns and 'nwound' in X_eng.columns:
        X_eng['total_casualties'] = X_eng[['nkill', 'nwound']].fillna(0).sum(axis=1)

    # Success binary
    if 'success' in X_eng.columns:
        X_eng['success_binary'] = X_eng['success'].fillna(0).astype(int)

    new_features = X_eng.shape[1] - initial_features
    print(f"Feature engineering complete. Added {new_features} new features")
    print(f"   Total features: {X_eng.shape[1]}")

    return X_eng


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """Perform stratified train-test split with handling of rare classes."""
    print("\n" + "=" * 60)
    print("BLOCK: TRAIN-TEST SPLIT")
    print("=" * 60)

    class_counts = y.value_counts()
    single_sample_classes = class_counts[class_counts < 2].index

    if len(single_sample_classes) > 0:
        # Remove classes with <2 samples (can't stratify)
        mask = ~y.isin(single_sample_classes)
        X = X[mask]
        y = y[mask]
        print(f"Removed {len(single_sample_classes)} classes with <2 samples")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Train: {X_train.shape[0]:,} samples; Test: {X_test.shape[0]:,} samples")
    return X_train, X_test, y_train, y_test


def balance_classes(X_train: pd.DataFrame, y_train: pd.Series, strategy: str = 'smote', random_state: int = 42) -> Tuple:
    """Balance classes using 'none', 'smote', or compute 'weights'."""
    print("\n" + "=" * 60)
    print("BLOCK: CLASS BALANCING")
    print("=" * 60)

    X_balanced = X_train
    y_balanced = y_train
    class_weights = None

    if strategy == 'none':
        print("No balancing applied")

    elif strategy == 'smote':
        print("Applying SMOTE oversampling to training set")
        sm = SMOTE(random_state=random_state)
        X_balanced, y_balanced = sm.fit_resample(X_train, y_train)

    elif strategy == 'weights':
        # compute simple inverse-frequency class weights
        counts = y_train.value_counts().to_dict()
        total = sum(counts.values())
        class_weights = {k: total / (len(counts) * v) for k, v in counts.items()}
        print(f"Computed class weights for {len(class_weights)} classes")
        print(f"   Weight range: {min(class_weights.values()):.3f} - {max(class_weights.values()):.3f}")

    else:
        raise ValueError(f"Unknown balancing strategy: {strategy}")

    print(f"Class balancing complete. Final shape: {getattr(X_balanced, 'shape', None)}")
    return X_balanced, y_balanced, class_weights


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Fit StandardScaler on X_train and transform both train and test."""
    print("\n" + "=" * 60)
    print("BLOCK: FEATURE SCALING")
    print("=" * 60)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Features standardized (mean=0, std=1)")
    print(f"   Training set: {X_train_scaled.shape}")
    print(f"   Test set: {X_test_scaled.shape}")

    return X_train_scaled, X_test_scaled, scaler


def run_preprocessing_pipeline(df: pd.DataFrame,
                               objective: int = 1,
                               missing_strategy: str = 'impute',
                               encoding_strategy: str = 'target',
                               balance_strategy: str = 'smote',
                               test_size: float = 0.2,
                               random_state: int = 42) -> Dict[str, Any]:
    """Run the complete preprocessing pipeline and return a dict of outputs."""
    print("\n" + "=" * 60)
    print("COMPLETE PREPROCESSING PIPELINE")
    print("=" * 60)

    # 1. Feature selection
    X, y, feature_names = select_features_objective1(df)

    # 2. Missing values
    X, y = handle_missing_values(X, y, strategy=missing_strategy)

    # 3. Encoding
    X = encode_categorical_features(X, y=y, strategy=encoding_strategy)

    # 4. Feature engineering
    X = engineer_features(X)

    # 5. Split
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, random_state=random_state)

    # 6. Balance
    X_train_bal, y_train_bal, class_weights = balance_classes(X_train, y_train, strategy=balance_strategy, random_state=random_state)

    # 7. Scale
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_bal, X_test)

    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train_bal.values if hasattr(y_train_bal, 'values') else y_train_bal,
        'y_test': y_test.values if hasattr(y_test, 'values') else y_test,
        'feature_names': feature_names,
        'class_weights': class_weights,
        'scaler': scaler
    }


def generate_experiment_combinations():
    """Return the cartesian product of preprocessing strategies (27 experiments)."""
    EXPERIMENT_CONFIG = {
        'missing_strategies': ['drop', 'impute', 'impute_flag'],
        'encoding_strategies': ['onehot', 'target', 'grouped'],
        'balance_strategies': ['none', 'smote', 'weights'],
    }

    combinations = list(product(
        EXPERIMENT_CONFIG['missing_strategies'],
        EXPERIMENT_CONFIG['encoding_strategies'],
        EXPERIMENT_CONFIG['balance_strategies']
    ))

    experiments = []
    for i, (miss, enc, bal) in enumerate(combinations, 1):
        exp_id = f"EXP{i:02d}"
        exp_name = f"{miss[:4].upper()}-{enc[:4].upper()}-{bal[:4].upper()}"
        experiments.append({
            'exp_id': exp_id,
            'exp_name': exp_name,
            'missing': miss,
            'encoding': enc,
            'balance': bal
        })

    return experiments
