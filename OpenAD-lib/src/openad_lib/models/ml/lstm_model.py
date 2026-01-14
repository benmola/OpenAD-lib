"""
LSTM-based surrogate model for anaerobic digestion time series prediction.

This module provides a PyTorch-based LSTM implementation for predicting biogas
production and other AD process outputs from feedstock composition inputs.

Example:
    >>> from openad_lib.models.ml import LSTMModel
    >>> model = LSTMModel(input_dim=5, hidden_dim=24, output_dim=1)
    >>> model.fit(X_train, y_train, epochs=50)
    >>> predictions = model.predict(X_test)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from typing import Optional, Tuple, List, Dict, Union
import warnings


def series_to_supervised(
    data: np.ndarray, 
    n_in: int = 1, 
    n_out: int = 1, 
    dropnan: bool = True
) -> pd.DataFrame:
    """
    Convert time series to supervised learning format.
    
    Args:
        data: Time series data as numpy array
        n_in: Number of lag observations as input (X)
        n_out: Number of observations as output (y)
        dropnan: Whether to drop rows with NaN values
    
    Returns:
        DataFrame framed for supervised learning
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = [], []
    
    # Input sequence (t-n, ..., t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(f'var{j+1}(t-{i})') for j in range(n_vars)]
    
    # Forecast sequence (t, t+1, ..., t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [(f'var{j+1}(t)') for j in range(n_vars)]
        else:
            names += [(f'var{j+1}(t+{i})') for j in range(n_vars)]
    
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    
    if dropnan:
        agg.dropna(inplace=True)
    
    return agg


class _LSTMNetwork(nn.Module):
    """PyTorch LSTM network architecture."""
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0
    ):
        super(_LSTMNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        # Take the last time step output
        lstm_out = lstm_out[:, -1, :]
        output = self.fc(lstm_out)
        return output


class LSTMModel:
    """
    LSTM-based surrogate model for AD process prediction.
    
    Uses PyTorch LSTM networks to learn temporal patterns in biogas
    production from feedstock composition time series.
    
    Attributes:
        input_dim: Number of input features
        hidden_dim: Number of LSTM hidden units
        output_dim: Number of output targets
        model: PyTorch LSTM network
        scaler_X: StandardScaler for input features
        scaler_y: StandardScaler for outputs
    
    Example:
        >>> model = LSTMModel(input_dim=5, hidden_dim=24, output_dim=1)
        >>> model.fit(X_train, y_train, epochs=100)
        >>> y_pred = model.predict(X_test)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 24,
        output_dim: int = 1,
        num_layers: int = 1,
        dropout: float = 0.0,
        learning_rate: float = 0.001,
        device: Optional[str] = None
    ):
        """
        Initialize LSTM model.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Number of hidden units in LSTM
            output_dim: Number of output targets
            num_layers: Number of LSTM layers
            dropout: Dropout probability (only applied if num_layers > 1)
            learning_rate: Optimizer learning rate
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        # Device configuration
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize network
        self.model = _LSTMNetwork(
            input_dim, hidden_dim, output_dim, num_layers, dropout
        ).to(self.device)
        
        # Scalers for normalization
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # Training state
        self.is_fitted = False
        self.training_history: List[float] = []
    
    def _prepare_data(
        self, 
        X: np.ndarray, 
        y: Optional[np.ndarray] = None,
        fit_scaler: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prepare data for LSTM: scale and convert to tensors."""
        # Scale X
        if fit_scaler:
            X_scaled = self.scaler_X.fit_transform(X)
        else:
            X_scaled = self.scaler_X.transform(X)
        
        # Reshape for LSTM: (samples, timesteps=1, features)
        X_tensor = torch.tensor(
            X_scaled.reshape(-1, 1, X_scaled.shape[1]), 
            dtype=torch.float32
        ).to(self.device)
        
        if y is not None:
            y_2d = y.reshape(-1, 1) if y.ndim == 1 else y
            if fit_scaler:
                y_scaled = self.scaler_y.fit_transform(y_2d)
            else:
                y_scaled = self.scaler_y.transform(y_2d)
            y_tensor = torch.tensor(y_scaled, dtype=torch.float32).to(self.device)
            return X_tensor, y_tensor
        
        return X_tensor, None
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 4,
        verbose: bool = True,
        validation_split: float = 0.0
    ) -> 'LSTMModel':
        """
        Train the LSTM model.
        
        Args:
            X: Input features array (n_samples, n_features)
            y: Target values array (n_samples,) or (n_samples, n_outputs)
            epochs: Number of training epochs
            batch_size: Mini-batch size
            verbose: Whether to print training progress
            validation_split: Fraction of data to use for validation
        
        Returns:
            self
        """
        # Prepare data
        X_tensor, y_tensor = self._prepare_data(X, y, fit_scaler=True)
        
        # Optimizer and loss
        criterion = nn.L1Loss()  # MAE loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.training_history = []
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            
            # Shuffle data
            permutation = torch.randperm(X_tensor.size(0))
            
            for batch_start in range(0, X_tensor.size(0), batch_size):
                indices = permutation[batch_start:batch_start + batch_size]
                batch_X = X_tensor[indices]
                batch_y = y_tensor[indices]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / (X_tensor.size(0) // batch_size + 1)
            self.training_history.append(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with trained model.
        
        Args:
            X: Input features array (n_samples, n_features)
        
        Returns:
            Predictions array (n_samples, n_outputs)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_tensor, _ = self._prepare_data(X, fit_scaler=False)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        # Inverse transform
        predictions = self.scaler_y.inverse_transform(predictions)
        
        return predictions
    
    def evaluate(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Test features
            y: True target values
        
        Returns:
            Dictionary with RMSE, MAE, and R² metrics
        """
        y_pred = self.predict(X)
        y_true = y.reshape(-1, 1) if y.ndim == 1 else y
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        return metrics
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        epochs: int = 50,
        batch_size: int = 4,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Perform time series cross-validation.
        
        Args:
            X: Input features array
            y: Target values array
            n_splits: Number of CV folds
            epochs: Training epochs per fold
            batch_size: Mini-batch size
            verbose: Print progress
        
        Returns:
            Dictionary with train and test metrics for each fold
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        results = {
            'train_rmse': [], 'test_rmse': [],
            'train_mae': [], 'test_mae': [],
            'train_r2': [], 'test_r2': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            if verbose:
                print(f"\n--- Fold {fold + 1}/{n_splits} ---")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Reset model
            self.model = _LSTMNetwork(
                self.input_dim, self.hidden_dim, self.output_dim,
                self.num_layers, self.dropout
            ).to(self.device)
            self.is_fitted = False
            
            # Train
            self.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=False)
            
            # Evaluate
            train_metrics = self.evaluate(X_train, y_train)
            test_metrics = self.evaluate(X_test, y_test)
            
            results['train_rmse'].append(train_metrics['rmse'])
            results['test_rmse'].append(test_metrics['rmse'])
            results['train_mae'].append(train_metrics['mae'])
            results['test_mae'].append(test_metrics['mae'])
            results['train_r2'].append(train_metrics['r2'])
            results['test_r2'].append(test_metrics['r2'])
            
            if verbose:
                print(f"Train RMSE: {train_metrics['rmse']:.4f}, Test RMSE: {test_metrics['rmse']:.4f}")
                print(f"Train R²: {train_metrics['r2']:.4f}, Test R²: {test_metrics['r2']:.4f}")
        
        return results
    
    def save(self, path: str) -> None:
        """Save model to file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim,
                'num_layers': self.num_layers,
                'dropout': self.dropout
            },
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'LSTMModel':
        """Load model from file."""
        checkpoint = torch.load(path)
        config = checkpoint['config']
        
        model = cls(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            output_dim=config['output_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        )
        
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.scaler_X = checkpoint['scaler_X']
        model.scaler_y = checkpoint['scaler_y']
        model.is_fitted = True
        
        return model
