import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    r2_score, mean_squared_error,
    accuracy_score, roc_auc_score,
    classification_report
)


def prepare_splits(X, y_reg, y_clf, test_size=0.2, random_state=42):
    """
    Prepare train/test splits and scaled features for both
    regression (pIC50) and classification (active/inactive) tasks.

    Returns:
        X_train_sc, X_test_sc       : scaled regression splits
        X_train_csc, X_test_csc     : scaled classification splits
        y_train, y_test             : regression targets
        y_train_c, y_test_c         : classification targets
        scaler                      : fitted StandardScaler
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_reg, test_size=test_size, random_state=random_state
    )
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_clf, test_size=test_size, random_state=random_state,
        stratify=y_clf
    )

    scaler = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    X_test_sc   = scaler.transform(X_test)
    X_train_csc = scaler.fit_transform(X_train_c)
    X_test_csc  = scaler.transform(X_test_c)

    return (X_train_sc, X_test_sc,
            X_train_csc, X_test_csc,
            y_train, y_test,
            y_train_c, y_test_c,
            scaler)


# Classical Regression Models

def train_ridge(X_train, X_test, y_train, y_test, alpha=1.0):
    """Ridge regression with 5-fold cross-validation."""
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    cv   = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    print(f"Ridge Regression → R²: {r2:.3f} | RMSE: {rmse:.3f}")
    print(f"  5-fold CV R²: {cv.mean():.3f} ± {cv.std():.3f}")
    return model, r2, rmse


def train_lasso(X_train, X_test, y_train, y_test, alpha=0.01):
    """Lasso regression — drives uninformative features to zero."""
    model = Lasso(alpha=alpha, max_iter=5000)
    model.fit(X_train, y_train)
    y_pred    = model.predict(X_test)
    r2        = r2_score(y_test, y_pred)
    n_nonzero = np.sum(model.coef_ != 0)
    print(f"Lasso Regression → R²: {r2:.3f} | Non-zero features: {n_nonzero}")
    return model, r2


def train_elasticnet(X_train, X_test, y_train, y_test,
                     alpha=0.01, l1_ratio=0.5):
    """ElasticNet — combines Ridge (L2) and Lasso (L1) regularisation."""
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    print(f"ElasticNet → R²: {r2:.3f}")
    return model, r2


# SVM Models

def train_svm_classifier(X_train, X_test, y_train, y_test):
    """SVM classifier (RBF kernel) for active/inactive prediction."""
    model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print(f"SVM Classifier → Accuracy: {acc:.3f} | AUC-ROC: {auc:.3f}")
    print(classification_report(y_test, y_pred,
                                 target_names=['Inactive', 'Active']))
    return model, acc, auc


def train_svr(X_train, X_test, y_train, y_test, C=1.0, epsilon=0.1):
    """SVR (RBF kernel) for pIC50 regression."""
    model = SVR(kernel='rbf', C=C, epsilon=epsilon)
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    print(f"SVR → R²: {r2:.3f}")
    return model, r2


# Random Forest Models

def train_random_forest_classifier(X_train, X_test, y_train, y_test,
                                    n_estimators=200, random_state=42):
    """Random Forest classifier with feature importance output."""
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=None,
        min_samples_leaf=2, random_state=random_state, n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print(f"Random Forest → Accuracy: {acc:.3f} | AUC-ROC: {auc:.3f}")
    return model, acc, auc


def train_random_forest_regressor(X_train, X_test, y_train, y_test,
                                   n_estimators=200, random_state=42):
    """Random Forest regressor for pIC50 prediction."""
    model = RandomForestRegressor(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1
    )
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    print(f"RF Regressor → R²: {r2:.3f}")
    return model, r2


def plot_feature_importance(rf_model, feature_names, top_n=20,
                             save_path='results/figures/rf_feature_importance.png'):
    """Plot and save top N feature importances from Random Forest."""
    importances = pd.Series(rf_model.feature_importances_, index=feature_names)
    top_feats   = importances.nlargest(top_n)

    print(f"\nTop {top_n} most important features:")
    print(top_feats)

    fig, ax = plt.subplots(figsize=(10, 6))
    top_feats.sort_values().plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title("Random Forest Feature Importance — EGFR QSAR Model")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Feature importance plot saved to {save_path}")
    return top_feats
