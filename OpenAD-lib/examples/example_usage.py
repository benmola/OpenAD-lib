"""
Example: Using OpenAD-lib Models

This script demonstrates how to use the different models in OpenAD-lib:
1. ADM1Model - Mechanistic model for anaerobic digestion
2. LSTMModel - LSTM neural network for time series prediction
3. MultitaskGP - Multi-task Gaussian Process for multi-output prediction

Run this script after installing openad_lib:
    pip install -e .
"""

import numpy as np
import pandas as pd
import os

# Get the data directory path
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'src', 'openad_lib', 'data')

# =============================================================================
# Example 1: Using Feedstock Library
# =============================================================================
print("=" * 60)
print("Example 1: Feedstock Library")
print("=" * 60)

from openad_lib.feedstock import FeedstockLibrary, FeedstockDescriptor

# Initialize the library with built-in feedstocks
lib = FeedstockLibrary()

# List available feedstocks
print(f"Available feedstocks: {lib.list_feedstocks()}")

# Get a specific feedstock
maize = lib.get("Maize")
print(f"\nMaize properties:")
print(f"  TS: {maize.ts} kg/m³")
print(f"  VS: {maize.vs} g/kg TS")
print(f"  BMP: {maize.bmp} NL CH4/kg VS")
print(f"  Proteins: {maize.proteins} kg/m³")

# Create a co-digestion mixture
mixture = lib.create_mixture(
    feedstock_names=["Maize", "Chicken Litter"],
    ratios=[0.7, 0.3],
    mixture_name="Maize-Chicken Mix"
)
print(f"\nCo-digestion mixture (70% Maize, 30% Chicken Litter):")
print(f"  Weighted BMP: {mixture.bmp:.1f} NL CH4/kg VS")

# Calculate ADM1 inputs from feedstock
adm1_inputs = lib.calculate_adm1_inputs(maize, flow_rate=130)
print(f"\nADM1 inputs calculated from Maize at 130 m³/day:")
print(f"  S_ac (acetate): {adm1_inputs['S_ac']:.2f} kg COD/m³")
print(f"  X_pr (proteins): {adm1_inputs['X_pr']:.2f} kg COD/m³")


# =============================================================================
# Example 2: ADM1 Model Simulation
# =============================================================================
print("\n" + "=" * 60)
print("Example 2: ADM1 Model Simulation")
print("=" * 60)

from openad_lib.models.mechanistic import ADM1Model

# Check if data files exist
influent_path = os.path.join(DATA_DIR, 'sample_ADM1_influent_data.csv')
initial_path = os.path.join(DATA_DIR, 'sample_initial_state.csv')

if os.path.exists(influent_path) and os.path.exists(initial_path):
    print("\nInitializing ADM1 model...")
    model = ADM1Model()
    
    # Configure reactor parameters
    model.params.V_liq = 10000  # m³
    model.params.V_gas = 600    # m³
    
    print(f"  Reactor volume: {model.params.V_liq} m³")
    print(f"  Gas headspace: {model.params.V_gas} m³")
    
    # Load data
    print("\nLoading influent and initial state data...")
    model.load_data(
        influent_path=influent_path,
        initial_state_path=initial_path,
        influent_sheet="Influent_ADM1_COD_Based"
    )
    
    # Run simulation (short demo - first 10 days)
    print("\nRunning ADM1 simulation...")
    print("(This may take a minute for full simulation)")
    
    # For a quick demo, we'll just show the setup is working
    print("ADM1 model ready for simulation!")
    print("Call model.run() to execute full simulation")
    
    # Uncomment to run full simulation:
    # results = model.run(solver_method="BDF", verbose=True)
    # model.plot_results()
else:
    print("\nData files not found. Showing ADM1 model structure:")
    model = ADM1Model()
    print(f"  Number of state variables: {len(model.STATE_NAMES)}")
    print(f"  State variables: {model.STATE_NAMES[:10]}...")


# =============================================================================
# Example 3: LSTM Model for Biogas Prediction
# =============================================================================
print("\n" + "=" * 60)
print("Example 3: LSTM Model for Biogas Prediction")
print("=" * 60)

try:
    from openad_lib.models.ml import LSTMModel
    
    # Check for sample data
    lstm_data_path = os.path.join(DATA_DIR, 'sample_feedstock_timeseries.csv')
    
    if os.path.exists(lstm_data_path):
        # Load the data
        data = pd.read_csv(lstm_data_path).dropna()
        print(f"\nLoaded {len(data)} samples from biogas plant data")
        
        # Define features and target
        features = ['Maize', 'Wholecrop', 'Chicken Litter', 'Lactose', 'Apple Pomace']
        target = 'Total_Biogas'
        
        X = data[features].values
        y = data[target].values
        
        print(f"  Features: {features}")
        print(f"  Target: {target}")
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Create and train model
        print("\nTraining LSTM model...")
        lstm = LSTMModel(input_dim=len(features), hidden_dim=24, output_dim=1)
        lstm.fit(X_train, y_train, epochs=20, verbose=False)
        
        # Evaluate
        metrics = lstm.evaluate(X_test, y_test)
        print(f"\nLSTM Test Metrics:")
        print(f"  RMSE: {metrics['rmse']:.2f}")
        print(f"  MAE: {metrics['mae']:.2f}")
        print(f"  R²: {metrics['r2']:.3f}")
    else:
        print("\nDemonstrating LSTM model with synthetic data...")
        # Create synthetic data
        np.random.seed(42)
        X_demo = np.random.randn(100, 5)
        y_demo = X_demo.sum(axis=1) + np.random.randn(100) * 0.1
        
        lstm = LSTMModel(input_dim=5, hidden_dim=24, output_dim=1)
        print("LSTM model created successfully!")
        print(f"  Input dimension: {lstm.input_dim}")
        print(f"  Hidden dimension: {lstm.hidden_dim}")
        
except ImportError as e:
    print(f"\nNote: PyTorch not installed. Install with: pip install torch")
    print(f"Error: {e}")


# =============================================================================
# Example 4: Multi-Task GP for Multi-Output Prediction
# =============================================================================
print("\n" + "=" * 60)
print("Example 4: Multi-Task GP for Multi-Output Prediction")
print("=" * 60)

try:
    from openad_lib.models.ml import MultitaskGP
    
    # Check for sample data
    mtgp_data_path = os.path.join(DATA_DIR, 'sample_ad_process_data.csv')
    
    if os.path.exists(mtgp_data_path):
        # Load the data
        data = pd.read_csv(mtgp_data_path)
        print(f"\nLoaded {len(data)} samples from AD process data")
        
        # Show available columns
        print(f"  Columns: {list(data.columns)[:8]}...")
        
        # Setup MTGP model (configuration only for demo)
        print("\nMTGP model configuration:")
        print("  - Number of tasks: 3 (e.g., SCOD, VFA, Biogas)")
        print("  - Number of latents: 3 (shared latent functions)")
        print("  - Uncertainty quantification: Yes (95% CI)")
        
        mtgp = MultitaskGP(num_tasks=3, num_latents=3)
        print("\nMulti-Task GP model created!")
        print("Call mtgp.fit(X_train, Y_train) to train")
    else:
        print("\nDemonstrating MTGP with synthetic data...")
        mtgp = MultitaskGP(num_tasks=3, num_latents=3)
        print("Multi-Task GP model created!")
        print(f"  Number of tasks: {mtgp.num_tasks}")
        print(f"  Number of latents: {mtgp.num_latents}")
        
except ImportError as e:
    print(f"\nNote: GPyTorch not installed. Install with: pip install gpytorch")
    print(f"Error: {e}")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("OpenAD-lib Summary")
print("=" * 60)
print("""
OpenAD-lib provides a unified framework for AD modelling:

MECHANISTIC MODELS:
  from openad_lib.models.mechanistic import ADM1Model
  model = ADM1Model()
  model.load_data(influent_path, initial_path)
  results = model.run()

ML SURROGATE MODELS:
  from openad_lib.models.ml import LSTMModel, MultitaskGP
  lstm = LSTMModel(input_dim=5, hidden_dim=24)
  mtgp = MultitaskGP(num_tasks=3)

FEEDSTOCK CHARACTERIZATION:
  from openad_lib.feedstock import FeedstockLibrary
  lib = FeedstockLibrary()
  maize = lib.get("Maize")
  mixture = lib.create_mixture(["Maize", "Grass"], [0.6, 0.4])

For more examples, see the notebooks/ directory.
""")
