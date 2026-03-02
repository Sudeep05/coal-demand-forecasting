"""
lstm_model.py — LSTM (Long Short-Term Memory) model for coal demand forecasting.
Two LSTM layers (128, 64 units) with Dropout and Dense output.
Uses PyTorch backend for Apple Silicon compatibility.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

try:
    from src.config import (
        LSTM_MODEL_PATH, MODELS_DIR, REPORTS_DIR, LSTM_LOSS_PATH,
        LSTM_SEQUENCE_LENGTH, LSTM_LAYER_1_UNITS, LSTM_LAYER_2_UNITS,
        LSTM_DROPOUT_RATE, LSTM_EPOCHS, LSTM_BATCH_SIZE,
        LSTM_EARLY_STOP_PATIENCE,
        PROCESSED_TRAIN_FILE, PROCESSED_VAL_FILE, PROCESSED_TEST_FILE,
    )
    from src.logger import get_logger
except ImportError:
    from config import (  # type: ignore[no-redef]
        LSTM_MODEL_PATH, MODELS_DIR, REPORTS_DIR, LSTM_LOSS_PATH,
        LSTM_SEQUENCE_LENGTH, LSTM_LAYER_1_UNITS, LSTM_LAYER_2_UNITS,
        LSTM_DROPOUT_RATE, LSTM_EPOCHS, LSTM_BATCH_SIZE,
        LSTM_EARLY_STOP_PATIENCE,
        PROCESSED_TRAIN_FILE, PROCESSED_VAL_FILE, PROCESSED_TEST_FILE,
    )
    from logger import get_logger  # type: ignore[no-redef]

logger = get_logger("training")

# Determine device
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch LSTM Model
# ──────────────────────────────────────────────────────────────────────────────
class CoalLSTM(nn.Module):
    """Two-layer LSTM for coal demand regression."""

    def __init__(self, n_features: int, layer1_units: int = 128,
                 layer2_units: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=n_features, hidden_size=layer1_units,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(
            input_size=layer1_units, hidden_size=layer2_units,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(layer2_units, 32)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out[:, -1, :])   # last time-step
        out = self.relu(self.fc1(out))
        return self.fc_out(out).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────
def _create_sequences(data: np.ndarray, target: np.ndarray,
                      seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding-window sequences for LSTM input."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i: i + seq_length])
        y.append(target[i + seq_length])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def _make_loader(X: np.ndarray, y: np.ndarray,
                 batch_size: int, shuffle: bool = True) -> DataLoader:
    """Wrap numpy arrays in a PyTorch DataLoader."""
    ds = TensorDataset(
        torch.from_numpy(X).to(DEVICE),
        torch.from_numpy(y).to(DEVICE),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ──────────────────────────────────────────────────────────────────────────────
# Public training entry-point
# ──────────────────────────────────────────────────────────────────────────────
def train_lstm(train_df: pd.DataFrame, val_df: pd.DataFrame,
               test_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Train LSTM model and forecast on the test period.

    Args:
        train_df: Training data.
        val_df: Validation data.
        test_df: Test data.

    Returns:
        Dict with predictions, actuals, model, and timing info.
    """
    logger.info("=" * 50)
    logger.info("Training LSTM model (PyTorch)...")
    logger.info(f"Device: {DEVICE}")
    logger.info("=" * 50)

    start_time = time.time()

    try:
        torch.manual_seed(42)
        np.random.seed(42)

        # Feature columns (exclude date and target)
        feature_cols = [c for c in train_df.columns
                        if c not in ["date", "coal_consumption_tonnes"]]

        # Prepare data
        X_train = train_df[feature_cols].values.astype(np.float32)
        y_train = train_df["coal_consumption_tonnes"].values.astype(np.float32)

        X_val = val_df[feature_cols].values.astype(np.float32)
        y_val = val_df["coal_consumption_tonnes"].values.astype(np.float32)

        # Combine train + first-half of val for training; second-half of val for validation
        X_train_full = np.concatenate([X_train, X_val[:len(X_val)//2]])
        y_train_full = np.concatenate([y_train, y_val[:len(y_val)//2]])

        X_val_seq_raw = np.concatenate([X_train[-LSTM_SEQUENCE_LENGTH:], X_val[len(X_val)//2:]])
        y_val_seq_raw = np.concatenate([y_train[-LSTM_SEQUENCE_LENGTH:], y_val[len(y_val)//2:]])

        # For test: tail of train+val data to seed initial sequences
        all_X = np.concatenate([X_train, X_val])
        all_y = np.concatenate([y_train, y_val])
        X_test_raw = test_df[feature_cols].values.astype(np.float32)
        y_test = test_df["coal_consumption_tonnes"].values.astype(np.float32)

        # Create sequences
        X_train_seq, y_train_seq = _create_sequences(X_train_full, y_train_full, LSTM_SEQUENCE_LENGTH)
        X_val_seq, y_val_seq = _create_sequences(X_val_seq_raw, y_val_seq_raw, LSTM_SEQUENCE_LENGTH)

        combined_test_X = np.concatenate([all_X[-LSTM_SEQUENCE_LENGTH:], X_test_raw])
        combined_test_y = np.concatenate([all_y[-LSTM_SEQUENCE_LENGTH:], y_test])
        X_test_seq, y_test_seq = _create_sequences(combined_test_X, combined_test_y, LSTM_SEQUENCE_LENGTH)

        n_features = X_train_seq.shape[2]

        logger.info(f"LSTM input shape: {X_train_seq.shape}")
        logger.info(f"Sequence length: {LSTM_SEQUENCE_LENGTH}")
        logger.info(f"Features: {n_features}")
        logger.info(f"Hyperparameters: layers=[{LSTM_LAYER_1_UNITS}, {LSTM_LAYER_2_UNITS}], "
                     f"dropout={LSTM_DROPOUT_RATE}, epochs={LSTM_EPOCHS}, "
                     f"batch_size={LSTM_BATCH_SIZE}")

        # DataLoaders
        train_loader = _make_loader(X_train_seq, y_train_seq, LSTM_BATCH_SIZE, shuffle=True)
        val_loader = _make_loader(X_val_seq, y_val_seq, LSTM_BATCH_SIZE, shuffle=False) if len(X_val_seq) > 0 else None

        # Build model
        model = CoalLSTM(
            n_features=n_features,
            layer1_units=LSTM_LAYER_1_UNITS,
            layer2_units=LSTM_LAYER_2_UNITS,
            dropout=LSTM_DROPOUT_RATE,
        ).to(DEVICE)

        logger.info(f"Model architecture:\n{model}")
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total parameters: {total_params:,}")

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        # ── Training loop with early stopping ───────────────────────────────
        os.makedirs(MODELS_DIR, exist_ok=True)
        best_val_loss = float("inf")
        patience_counter = 0
        train_losses: List[float] = []
        val_losses: List[float] = []

        for epoch in range(1, LSTM_EPOCHS + 1):
            # -- train --
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            avg_train_loss = epoch_loss / max(n_batches, 1)
            train_losses.append(avg_train_loss)

            # -- validate --
            if val_loader is not None:
                model.eval()
                val_epoch_loss = 0.0
                val_n = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        preds = model(xb)
                        val_epoch_loss += criterion(preds, yb).item()
                        val_n += 1
                avg_val_loss = val_epoch_loss / max(val_n, 1)
                val_losses.append(avg_val_loss)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save best checkpoint
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "n_features": n_features,
                        "layer1_units": LSTM_LAYER_1_UNITS,
                        "layer2_units": LSTM_LAYER_2_UNITS,
                        "dropout": LSTM_DROPOUT_RATE,
                    }, LSTM_MODEL_PATH)
                else:
                    patience_counter += 1

                if epoch % 10 == 0 or epoch == 1:
                    logger.info(f"Epoch {epoch:3d}/{LSTM_EPOCHS}  "
                                f"train_loss={avg_train_loss:.4f}  "
                                f"val_loss={avg_val_loss:.4f}  "
                                f"patience={patience_counter}/{LSTM_EARLY_STOP_PATIENCE}")

                if patience_counter >= LSTM_EARLY_STOP_PATIENCE:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            else:
                val_losses.append(avg_train_loss)
                if epoch % 10 == 0 or epoch == 1:
                    logger.info(f"Epoch {epoch:3d}/{LSTM_EPOCHS}  "
                                f"train_loss={avg_train_loss:.4f}")

        # Restore best weights
        if os.path.exists(LSTM_MODEL_PATH):
            ckpt = torch.load(LSTM_MODEL_PATH, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            logger.info("Restored best model checkpoint")

        # ── Plot training loss ──────────────────────────────────────────────
        os.makedirs(REPORTS_DIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(train_losses, label="Training Loss", color="#E74C3C")
        if val_losses:
            ax.plot(val_losses, label="Validation Loss", color="#3498DB")
        ax.set_title("LSTM Training Loss Curve", fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(LSTM_LOSS_PATH, dpi=150)
        plt.close(fig)
        logger.info(f"LSTM loss curve saved to {LSTM_LOSS_PATH}")

        # ── Predict on test ─────────────────────────────────────────────────
        model.eval()
        X_test_tensor = torch.from_numpy(X_test_seq).to(DEVICE)
        inference_start = time.time()
        with torch.no_grad():
            predictions = model(X_test_tensor).cpu().numpy()
        inference_time = (time.time() - inference_start) * 1000

        training_time = time.time() - start_time
        actuals = y_test_seq

        logger.info(f"LSTM training completed in {training_time:.2f}s")
        logger.info(f"LSTM inference time: {inference_time:.2f}ms for {len(predictions)} samples")

        return {
            "model_name": "LSTM",
            "predictions": predictions,
            "actuals": actuals,
            "model": model,
            "training_time": training_time,
            "inference_time_ms": inference_time,
            "history": {"loss": train_losses, "val_loss": val_losses},
        }

    except Exception as e:
        logger.error(f"LSTM training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    train_df = pd.read_csv(PROCESSED_TRAIN_FILE)
    val_df = pd.read_csv(PROCESSED_VAL_FILE)
    test_df = pd.read_csv(PROCESSED_TEST_FILE)
    results = train_lstm(train_df, val_df, test_df)
    print(f"LSTM predictions shape: {results['predictions'].shape}")
