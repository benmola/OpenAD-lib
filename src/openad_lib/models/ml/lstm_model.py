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
from pathlib import Path
import warnings

from openad_lib.models.base import MLModel


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


class LSTMModel(MLModel):
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
        super().__init__(params=None)
        
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
        
        # Base class compatibility
        self.results: Optional[Dict] = None
        self.metrics: Optional[Dict[str, float]] = None

    @staticmethod
    def series_to_supervised(
        data: Union[np.ndarray, pd.DataFrame], 
        n_in: int = 1, 
        n_out: int = 1, 
        dropnan: bool = True
    ) -> pd.DataFrame:
        """
        Convert time series to supervised learning format (static utility).
        
        Args:
            data: Time series data
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

    def prepare_time_series_data(
        self, 
        data: pd.DataFrame, 
        features: List[str], 
        target: str,
        n_in: int = 1,
        n_out: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Preprocess time series data for training.
        
        Helper method that handles:
        1. Feature selection
        2. Supervised learning transformation (lags)
        3. Separation into X and y
        
        Args:
            data: Raw DataFrame
            features: List of feature column names
            target: Target column name
            n_in: Number of lag steps (input window)
            n_out: Number of forecast steps (horizon)
            
        Returns:
            X: Input features array
            y: Target array
            reframed: The processed DataFrame
        """
        # Ensure we have the data
        if not all(col in data.columns for col in features + [target]):
            missing = [c for c in features + [target] if c not in data.columns]
            raise ValueError(f"Missing columns: {missing}")
            
        # Select columns
        values = data[features].values.astype('float32')
        target_vals = data[[target]].values.astype('float32')
        
        # Use series_to_supervised on features
        # 1. Create Lagged Features
        df_features = pd.DataFrame(values, columns=features)
        cols, names = [], []
        for i in range(n_in, 0, -1):
            cols.append(df_features.shift(i))
            names += [f'{col}(t-{i})' for col in features]
        
        X_df = pd.concat(cols, axis=1)
        X_df.columns = names
        
        # 2. Target
        y_df = pd.DataFrame(target_vals, columns=[target])
        
        # 3. Concatenate and Drop NaNs
        dataset = pd.concat([X_df, y_df], axis=1)
        dataset.dropna(inplace=True)
        
        # 4. Split
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        
        return X, y, dataset
    
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
        
        # Use unified metrics from utils
        from openad_lib.utils.metrics import compute_metrics
        self.metrics = compute_metrics(y_true.flatten(), y_pred.flatten())
        
        return self.metrics
    
    # ========== Base Class Interface Methods ==========
    
    def load_data(self, filepath: Path) -> pd.DataFrame:
        """
        Load time series data from file (implements BaseModel.load_data).
        
        Parameters
        ----------
        filepath : str or Path
            Path to CSV file
            
        Returns
        -------
        data : pd.DataFrame
            Loaded dataframe
        """
        self.data = pd.read_csv(filepath)
        return self.data
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ):
        """
        Train the model (implements MLModel.train).
        
        This is an alias for fit() to comply with base class interface.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training targets
        X_val : np.ndarray, optional
            Validation features (not used currently)
        y_val : np.ndarray, optional
            Validation targets (not used currently)
        **kwargs
            Additional training options (epochs, batch_size, verbose)
        """
        return self.fit(X_train, y_train, **kwargs)
