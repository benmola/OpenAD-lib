"""
Multi-Task Gaussian Process (MTGP) model for anaerobic digestion.

This module provides a GPyTorch-based multi-task GP implementation using
Linear Model of Coregionalization (LMC) for predicting multiple AD outputs
simultaneously (e.g., SCOD, VFA, Biogas) with uncertainty quantification.

Example:
    >>> from openad_lib.models.ml import MultitaskGP
    >>> model = MultitaskGP(num_tasks=3, num_latents=3)
    >>> model.fit(X_train, Y_train, epochs=100)
    >>> mean, lower, upper = model.predict(X_test)
"""

import torch
import gpytorch
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict, Union
from sklearn.preprocessing import StandardScaler


class _MultitaskGPModelLMC(gpytorch.models.ApproximateGP):
    """
    GPyTorch Multi-Task GP using Linear Model of Coregionalization.
    
    Uses variational inference for scalability with inducing points.
    """
    
    def __init__(
        self,
        inducing_points: torch.Tensor,
        num_tasks: int,
        num_latents: int
    ):
        # Variational distribution for latent functions
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2),
            batch_shape=torch.Size([num_latents])
        )
        
        # Variational strategy with LMC
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True
            ),
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1
        )
        
        super().__init__(variational_strategy)
        
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([num_latents])
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])),
            batch_shape=torch.Size([num_latents])
        )
    
    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MultitaskGP:
    """
    Multi-Task Gaussian Process for multi-output AD prediction.
    
    Uses GPyTorch with Linear Model of Coregionalization (LMC) to model
    correlations between multiple outputs (e.g., SCOD, VFA, Biogas) while
    providing uncertainty quantification for each prediction.
    
    Attributes:
        num_tasks: Number of output tasks
        num_latents: Number of latent functions
        model: GPyTorch MTGP model
        likelihood: Multitask Gaussian likelihood
    
    Example:
        >>> model = MultitaskGP(num_tasks=3, num_latents=3)
        >>> model.fit(X_train, Y_train)
        >>> mean, lower, upper = model.predict(X_test, return_std=True)
    """
    
    def __init__(
        self,
        num_tasks: int = 3,
        num_latents: int = 3,
        n_inducing: int = 60,
        learning_rate: float = 0.1,
        log_transform: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize Multi-Task GP model.
        
        Args:
            num_tasks: Number of output tasks to predict
            num_latents: Number of latent functions for LMC
            n_inducing: Number of inducing points for variational inference
            learning_rate: Optimizer learning rate
            log_transform: Whether to apply log transform to outputs
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.num_tasks = num_tasks
        self.num_latents = num_latents
        self.n_inducing = n_inducing
        self.learning_rate = learning_rate
        self.log_transform = log_transform
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model: Optional[_MultitaskGPModelLMC] = None
        self.likelihood: Optional[gpytorch.likelihoods.MultitaskGaussianLikelihood] = None
        
        # Scalers
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        self.is_fitted = False
        self.training_losses: List[float] = []
    
    def _prepare_data(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        fit_scaler: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prepare data: scale and convert to tensors."""
        if fit_scaler:
            X_scaled = self.scaler_X.fit_transform(X)
        else:
            X_scaled = self.scaler_X.transform(X)
        
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        if Y is not None:
            if self.log_transform:
                Y_transformed = np.log(Y + 1e-6)
            else:
                Y_transformed = Y
            
            if fit_scaler:
                Y_scaled = self.scaler_y.fit_transform(Y_transformed)
            else:
                Y_scaled = self.scaler_y.transform(Y_transformed)
            
            Y_tensor = torch.tensor(Y_scaled, dtype=torch.float32).to(self.device)
            return X_tensor, Y_tensor
        
        return X_tensor, None
    
    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        epochs: int = 100,
        verbose: bool = True
    ) -> 'MultitaskGP':
        """
        Train the Multi-Task GP model.
        
        Args:
            X: Input features (n_samples, n_features)
            Y: Target outputs (n_samples, num_tasks)
            epochs: Number of training epochs
            verbose: Whether to print progress
        
        Returns:
            self
        """
        X_tensor, Y_tensor = self._prepare_data(X, Y, fit_scaler=True)
        
        # Select inducing points from training data
        n_inducing = min(self.n_inducing, X_tensor.size(0))
        inducing_points = X_tensor[:n_inducing].unsqueeze(0).repeat(
            self.num_latents, 1, 1
        )
        
        # Initialize model and likelihood
        self.model = _MultitaskGPModelLMC(
            inducing_points, self.num_tasks, self.num_latents
        ).to(self.device)
        
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=self.num_tasks
        ).to(self.device)
        
        # Training mode
        self.model.train()
        self.likelihood.train()
        
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()}
        ], lr=self.learning_rate)
        
        mll = gpytorch.mlls.VariationalELBO(
            self.likelihood, self.model, num_data=Y_tensor.size(0)
        )
        
        self.training_losses = []
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.model(X_tensor)
            loss = -mll(output, Y_tensor)
            loss.backward()
            optimizer.step()
            
            self.training_losses.append(loss.item())
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        
        self.is_fitted = True
        return self
    
    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Make predictions with uncertainty quantification.
        
        Args:
            X: Input features (n_samples, n_features)
            return_std: If True, return (mean, lower, upper) confidence bounds
        
        Returns:
            If return_std=True: (mean, lower_bound, upper_bound) arrays
            If return_std=False: mean predictions array
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_tensor, _ = self._prepare_data(X, fit_scaler=False)
        
        # Prediction mode
        self.model.eval()
        self.likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(self.model(X_tensor))
            mean = predictions.mean
            
            if return_std:
                lower, upper = predictions.confidence_region()
        
        # Convert to numpy and inverse transform
        mean_np = mean.cpu().numpy()
        mean_np = self.scaler_y.inverse_transform(mean_np)
        
        if self.log_transform:
            mean_np = np.exp(mean_np)
        
        if return_std:
            lower_np = self.scaler_y.inverse_transform(lower.cpu().numpy())
            upper_np = self.scaler_y.inverse_transform(upper.cpu().numpy())
            
            if self.log_transform:
                lower_np = np.exp(lower_np)
                upper_np = np.exp(upper_np)
            
            return mean_np, lower_np, upper_np
        
        return mean_np
    
    def evaluate(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        task_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model performance for each task.
        
        Args:
            X: Test features
            Y: True target values (n_samples, num_tasks)
            task_names: Optional list of task names for output dict
        
        Returns:
            Dictionary with metrics per task
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mean_pred = self.predict(X, return_std=False)
        
        if task_names is None:
            task_names = [f'Task_{i}' for i in range(self.num_tasks)]
        
        metrics = {}
        for i, name in enumerate(task_names):
            metrics[name] = {
                'rmse': np.sqrt(mean_squared_error(Y[:, i], mean_pred[:, i])),
                'mae': mean_absolute_error(Y[:, i], mean_pred[:, i]),
                'r2': r2_score(Y[:, i], mean_pred[:, i])
            }
        
        return metrics
    
    def plot_predictions(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        task_names: Optional[List[str]] = None,
        x_label: str = "Time",
        figsize: Tuple[int, int] = (15, 10)
    ) -> None:
        """
        Plot training and test predictions with uncertainty.
        
        Args:
            X_train, Y_train: Training data
            X_test, Y_test: Test data
            task_names: Names for each output task
            x_label: Label for x-axis
            figsize: Figure size
        """
        import matplotlib.pyplot as plt
        
        if task_names is None:
            task_names = [f'Task_{i}' for i in range(self.num_tasks)]
        
        # Get predictions
        train_mean, train_lower, train_upper = self.predict(X_train, return_std=True)
        test_mean, test_lower, test_upper = self.predict(X_test, return_std=True)
        
        fig, axes = plt.subplots(2, self.num_tasks, figsize=figsize)
        
        for i in range(self.num_tasks):
            # Training
            axes[0, i].plot(X_train[:, 0], Y_train[:, i], 'bo', label='True', markersize=5)
            axes[0, i].plot(X_train[:, 0], train_mean[:, i], 'r-', label='Predicted')
            axes[0, i].fill_between(
                X_train[:, 0], train_lower[:, i], train_upper[:, i],
                alpha=0.3, color='red', label='95% CI'
            )
            axes[0, i].set_title(f'{task_names[i]} (Training)')
            axes[0, i].set_xlabel(x_label)
            axes[0, i].legend()
            
            # Testing
            axes[1, i].plot(X_test[:, 0], Y_test[:, i], 'go', label='True', markersize=5)
            axes[1, i].plot(X_test[:, 0], test_mean[:, i], 'r-', label='Predicted')
            axes[1, i].fill_between(
                X_test[:, 0], test_lower[:, i], test_upper[:, i],
                alpha=0.3, color='red', label='95% CI'
            )
            axes[1, i].set_title(f'{task_names[i]} (Testing)')
            axes[1, i].set_xlabel(x_label)
            axes[1, i].legend()
        
        plt.tight_layout()
        plt.show()
    
    def save(self, path: str) -> None:
        """Save model to file."""
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'likelihood_state_dict': self.likelihood.state_dict(),
            'config': {
                'num_tasks': self.num_tasks,
                'num_latents': self.num_latents,
                'n_inducing': self.n_inducing,
                'log_transform': self.log_transform
            },
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y
        }, path)
    
    @classmethod
    def load(cls, path: str, X_sample: np.ndarray) -> 'MultitaskGP':
        """
        Load model from file.
        
        Args:
            path: Path to saved model
            X_sample: Sample input for initializing inducing points shape
        
        Returns:
            Loaded MultitaskGP model
        """
        checkpoint = torch.load(path)
        config = checkpoint['config']
        
        model = cls(
            num_tasks=config['num_tasks'],
            num_latents=config['num_latents'],
            n_inducing=config['n_inducing'],
            log_transform=config['log_transform']
        )
        
        model.scaler_X = checkpoint['scaler_X']
        model.scaler_y = checkpoint['scaler_y']
        
        # Need to initialize model structure
        X_scaled = model.scaler_X.transform(X_sample)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(model.device)
        n_inducing = min(model.n_inducing, X_tensor.size(0))
        inducing_points = X_tensor[:n_inducing].unsqueeze(0).repeat(
            model.num_latents, 1, 1
        )
        
        model.model = _MultitaskGPModelLMC(
            inducing_points, model.num_tasks, model.num_latents
        ).to(model.device)
        model.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=model.num_tasks
        ).to(model.device)
        
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
        model.is_fitted = True
        
        return model
