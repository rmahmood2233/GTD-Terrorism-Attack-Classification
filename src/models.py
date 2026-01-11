"""Model training and evaluation helpers."""
from typing import Optional, Dict, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def train_random_forest(X_train, y_train, class_weights: Optional[Dict] = None, random_state: int = 42):
    print("\n" + "=" * 60)
    print("BLOCK: TRAIN RANDOM FOREST")
    print("=" * 60)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=random_state,
        class_weight='balanced' if class_weights else None,
        n_jobs=-1,
        verbose=0,
    )

    print("Training Random Forest...")
    model.fit(X_train, y_train)
    print("Random Forest training completed!")
    return model


def train_logistic_regression(X_train, y_train, class_weights: Optional[Dict] = None, random_state: int = 42):
    print("\n" + "=" * 60)
    print("BLOCK: TRAIN LOGISTIC REGRESSION")
    print("=" * 60)

    model = LogisticRegression(
        solver='lbfgs',
        multi_class='multinomial',
        max_iter=1000,
        class_weight='balanced' if class_weights else None,
    )

    print("Training Logistic Regression...")
    model.fit(X_train, y_train)
    print("Logistic Regression training completed!")
    return model


def train_gradient_boosting(X_train, y_train, random_state: int = 42):
    print("\n" + "=" * 60)
    print("BLOCK: TRAIN GRADIENT BOOSTING")
    print("=" * 60)

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        random_state=random_state,
    )

    print("Training Gradient Boosting...")
    model.fit(X_train, y_train)
    print("Gradient Boosting training completed!")
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name: str = "Model") -> Dict[str, Any]:
    """Compute common evaluation metrics and return results dict."""
    print(f"\nEvaluating {model_name}...")

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    results = {
        'model_name': model_name,
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'train_macro_f1': f1_score(y_train, y_train_pred, average='macro'),
        'test_macro_f1': f1_score(y_test, y_test_pred, average='macro'),
        'classification_report': classification_report(y_test, y_test_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_test_pred),
    }

    overfit_score = results['train_accuracy'] - results['test_accuracy']
    if overfit_score > 0.1:
        print(f"Potential overfitting detected! (Gap: {overfit_score:.4f})")
    else:
        print(f"Model generalization looks good! (Gap: {overfit_score:.4f})")

    return results