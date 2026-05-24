import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


# ─── PyTorch Neural Network ───────────────────────────────────────────────────

class ADMETNet(nn.Module):
    """
    Feedforward neural network for pIC50 regression.
    Architecture: Input → 512 → 256 → 128 → 1
    BatchNorm stabilises training on molecular descriptors.
    Dropout prevents overfitting on small compound datasets.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 1)   # Single output: pIC50
        )

    def forward(self, x):
        return self.network(x)


def train_pytorch_nn(X, y_reg, test_size=0.2, random_state=42,
                     epochs=60, batch_size=64, lr=1e-3,
                     save_plot_path='results/figures/nn_training_loss.png'):
    """
    Full PyTorch training pipeline for pIC50 regression.

    Steps:
    - Train/test split + StandardScaler
    - Convert to PyTorch tensors
    - Train ADMETNet with Adam + StepLR scheduler
    - Evaluate on test set
    - Save training loss plot

    Returns:
        model    : trained ADMETNet
        r2       : R² on test set
        rmse     : RMSE on test set
        losses   : list of per-epoch training losses
    """
    # Split + scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_reg, test_size=test_size, random_state=random_state
    )
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Tensors + DataLoader
    X_tr_t = torch.FloatTensor(X_train_sc)
    y_tr_t = torch.FloatTensor(y_train).view(-1, 1)
    X_te_t = torch.FloatTensor(X_test_sc)

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Model + optimiser
    input_dim = X_train_sc.shape[1]
    model     = ADMETNet(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Training loop
    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_dl:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / len(train_dl)
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(X_te_t).numpy().flatten()

    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"\nPyTorch NN → R²: {r2:.3f} | RMSE: {rmse:.3f}")

    # Save training loss plot
    os.makedirs(os.path.dirname(save_plot_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(losses)
    ax.set_title("PyTorch Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    plt.tight_layout()
    plt.savefig(save_plot_path, dpi=150)
    print(f"Training loss plot saved to {save_plot_path}")

    return model, r2, rmse, losses


# ─── Model Comparison Table ───────────────────────────────────────────────────

def build_comparison_table(metrics: dict,
                            save_dir='results/'):
    """
    Build and save regression + classification model comparison tables.

    Args:
        metrics : dict with keys:
                  r2_ridge, rmse_ridge, r2_lasso, r2_en, r2_svr,
                  r2_rf_reg, r2_nn, rmse_nn,
                  acc_svm, auc_svm, acc_rf, auc_rf
        save_dir: directory to save CSV files

    Returns:
        reg_df  : regression results DataFrame
        clf_df  : classification results DataFrame
    """
    reg_df = pd.DataFrame({
        'Model': ['PyTorch NN', 'SVR', 'ElasticNet',
                  'Ridge Regression', 'Lasso', 'RF Regressor'],
        'Task':  ['Regression'] * 6,
        'R²':    [metrics['r2_nn'],    metrics['r2_svr'],
                  metrics['r2_en'],    metrics['r2_ridge'],
                  metrics['r2_lasso'], metrics['r2_rf_reg']],
        'RMSE':  [metrics['rmse_nn'],  np.nan, np.nan,
                  metrics['rmse_ridge'], np.nan, np.nan]
    }).sort_values('R²', ascending=False)

    clf_df = pd.DataFrame({
        'Model':    ['SVM Classifier', 'RF Classifier'],
        'Task':     ['Classification'] * 2,
        'Accuracy': [metrics['acc_svm'], metrics['acc_rf']],
        'AUC-ROC':  [metrics['auc_svm'], metrics['auc_rf']]
    }).sort_values('AUC-ROC', ascending=False)

    os.makedirs(save_dir, exist_ok=True)
    reg_df.to_csv(os.path.join(save_dir, 'model_comparison.csv'), index=False)
    clf_df.to_csv(os.path.join(save_dir, 'classification_model_comparison.csv'),
                  index=False)

    print("\nRegression Models:")
    print(reg_df.to_string(index=False))
    print("\nClassification Models:")
    print(clf_df.to_string(index=False))

    return reg_df, clf_df
