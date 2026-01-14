"""
LSTM Prediction Example
=======================

This script demonstrates LSTM-based biogas prediction using openad_lib,
with data preprocessing matching the reference implementation.

Features:
- Uses series_to_supervised for time-lagged features
- 80/20 train/test split
- Uses openad_lib.models.ml.LSTMModel

Usage:
    python examples/04_lstm_prediction.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from openad_lib.models.ml import LSTMModel
except ImportError as e:
    print(f"Error importing openad_lib: {e}")
    sys.exit(1)

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """Convert series to supervised learning format."""
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = [], []
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def main():
    print("="*60)
    print("LSTM Model for Biogas Prediction")
    print("="*60)

    # Paths
    base_dir = os.path.dirname(__file__)
    project_root = os.path.join(base_dir, '..')
    data_path = os.path.join(project_root, 'src', 'openad_lib', 'data', 'sample_LSTM_timeseries.csv')

    # Load data
    if os.path.exists(data_path):
        data = pd.read_csv(data_path).dropna()
        print(f"\nLoaded {len(data)} samples from biogas plant data")
        
        # Define features and target (matching reference)
        features = ['Maize', 'Wholecrop', 'Chicken Litter', 'Lactose', 'Apple Pomace', 'Rice bran']
        label = 'Total_Biogas'
        
        print(f"  Features: {features}")
        print(f"  Target: {label}")
    else:
        print("Error: Data file not found")
        return

    # Prepare data for normalization (matching reference)
    values = data[features].values.astype('float32')
    scaler = StandardScaler()
    scaled = scaler.fit_transform(values)

    # Frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)

    # Scale the output variable
    y = data[[label]].values
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)

    # 80/20 Split (matching user request)
    split_idx = int(len(reframed) * 0.8)
    train = reframed.values[:split_idx]
    test = reframed.values[split_idx:]
    
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]

    print(f"\nTraining samples: {len(train_X)}")
    print(f"Testing samples: {len(test_X)}")

    # Create and train model using openad_lib
    print("\nTraining LSTM model...")
    # Note: input_dim must match the actual feature count after series_to_supervised
    lstm = LSTMModel(input_dim=train_X.shape[1], hidden_dim=24, output_dim=1)
    
    # Note: openad_lib's fit expects unscaled data, but we need to use pre-scaled
    # So we'll use the model's internal methods directly
    lstm.fit(train_X, train_y, epochs=50, verbose=True)

    # Predict
    trainPredict = lstm.predict(train_X)
    testPredict = lstm.predict(test_X)

    # Inverse transform
    trainPredict = scaler_y.inverse_transform(trainPredict)
    testPredict = scaler_y.inverse_transform(testPredict)
    train_y_inv = scaler_y.inverse_transform(train_y.reshape(-1, 1))
    test_y_inv = scaler_y.inverse_transform(test_y.reshape(-1, 1))

    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(train_y_inv, trainPredict))
    test_rmse = np.sqrt(mean_squared_error(test_y_inv, testPredict))
    train_mae = mean_absolute_error(train_y_inv, trainPredict)
    test_mae = mean_absolute_error(test_y_inv, testPredict)
    train_r2 = r2_score(train_y_inv, trainPredict)
    test_r2 = r2_score(test_y_inv, testPredict)

    print(f"\nLSTM Metrics:")
    print(f"Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}")
    print(f"Train MAE: {train_mae:.2f}, Test MAE: {test_mae:.2f}")
    print(f"Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}")

    # Plotting
    print("\nGenerating prediction plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_y_inv, label='Actual', color='blue', alpha=0.7)
    ax1.plot(trainPredict, label='Predicted', color='red', linestyle='--')
    ax1.set_title("Training Set")
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel("Biogas Production")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(test_y_inv, label='Actual', color='blue', alpha=0.7)
    ax2.plot(testPredict, label='Predicted', color='red', linestyle='--')
    ax2.set_title("Testing Set")
    ax2.set_xlabel("Sample Index")
    ax2.set_ylabel("Biogas Production")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    images_dir = os.path.join(project_root, 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    save_path = os.path.join(images_dir, 'lstm_prediction_result.png')
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    main()
