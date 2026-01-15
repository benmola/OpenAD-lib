# OpenAD-lib

**Open-Source Framework for Uncertainty-Aware Anaerobic Digestion Digital Twins**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

OpenAD-lib is a unified Python library for anaerobic digestion (AD) process modelling, control, and scheduling with explicit uncertainty quantification. It provides both mechanistic models (ADM1, AM2) and machine learning surrogates (LSTM, Multi-Task GP) for biogas production prediction.

## 🚀 Features

- **Mechanistic Models**: Complete ADM1 (38 state variables) and simplified AM2 (4 state variables)
- **ML Surrogate Models**: LSTM and Multi-Task Gaussian Process with uncertainty quantification
- **Model Calibration**: Optuna-based parameter optimization for AM2 model
- **Model Predictive Control**: do-mpc based MPC for biogas optimization and VFA tracking
- **Feedstock Library**: Built-in database of 12 common AD substrates
- **ACoD Preprocessing**: Automatic influent characterization from feedstock ratios
- **Uncertainty Propagation**: From feedstock variability through model predictions

## 📦 Installation

### Using pip (recommended)

```bash
# Clone the repository
git clone https://github.com/benmola/OpenAD-lib.git
cd OpenAD-lib

# Install in development mode
pip install -e .

# For ML models (LSTM, Gaussian Process)
pip install -e ".[ml]"

# For optimization features (Optuna)
pip install -e ".[optimization]"

# For MPC control (do-mpc)
pip install -e ".[control]"

# For all features
pip install -e ".[full]"
```

### Using conda

```bash
# Create a new environment
conda create -n openad python=3.10
conda activate openad

# Install dependencies
pip install -e ".[full]"
```

## 🏃 Quick Start

### 1. ADM1 Simulation with ACoD Preprocessing

```python
from openad_lib.preprocessing import acod
from openad_lib.models.mechanistic import ADM1Model

# Generate influent data from feedstock ratios
influent_df = acod.generate_influent_data("path/to/feed_ratios.csv")

# Initialize and run ADM1 simulation
model = ADM1Model()
results = model.simulate(influent_df)

# Access results
biogas = results['q_gas']
states = results['results']
```

### 2. AM2 Simplified Model

```python
from openad_lib.models.mechanistic import AM2Model

# Initialize model with default parameters
model = AM2Model()

# Load experimental data
model.load_data("path/to/lab_data.csv")

# Run simulation and evaluate
results = model.run(verbose=True)
model.print_metrics()
model.plot_results()
```

### 3. AM2 Parameter Calibration

```python
from openad_lib.optimisation import AM2Calibrator
from openad_lib.models.mechanistic import AM2Model

# Initialize model
model = AM2Model()
model.load_data("path/to/lab_data.csv")

# Configure calibrator
calibrator = AM2Calibrator(model)

# Define custom parameter bounds (optional)
custom_bounds = {
    'm1': (0.05, 0.5),    # Growth rate bounds
    'K1': (5.0, 30.0),    # Half-saturation bounds
    'm2': (0.1, 1.0),
    'Ki': (10.0, 50.0),
    'K2': (20.0, 80.0)
}

# Run optimization with custom bounds
best_params = calibrator.calibrate(
    params_to_tune=['m1', 'K1', 'm2', 'Ki', 'K2'],
    param_bounds=custom_bounds,  # Optional: use default bounds if not specified
    n_trials=100,
    weights={'S1': 0.5, 'S2': 1.0, 'Q': 1.0}
)
```

### 4. Model Predictive Control (MPC)

```python
from openad_lib.control import AM2MPCController
from openad_lib.models.mechanistic import AM2Model

# Initialize model and controller
model = AM2Model()
controller = AM2MPCController(model)

# Configure MPC
controller.setup_controller(
    objective='maximize_biogas',
    prediction_horizon=10,
    control_horizon=5
)

# Run control loop
for t in range(simulation_time):
    control_action = controller.compute_control(current_state)
    new_state = model.step(control_action)
```

### 5. LSTM Surrogate Model

```python
from openad_lib.models.ml import LSTMModel

# Create and train LSTM model
lstm = LSTMModel(input_dim=6, hidden_dim=24, output_dim=1)
lstm.fit(X_train, y_train, epochs=50, batch_size=4)

# Predict and evaluate
predictions = lstm.predict(X_test)
metrics = lstm.evaluate(X_test, y_test)
print(f"Test RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.3f}")
```

### 6. Multi-Task GP with Uncertainty

```python
from openad_lib.models.ml import MultitaskGP

# Predict multiple outputs (SCOD, VFA, Biogas) with uncertainty
mtgp = MultitaskGP(num_tasks=3, num_latents=3, log_transform=True)
mtgp.fit(X_train, Y_train, epochs=500)

# Get predictions with 95% confidence intervals
mean, lower, upper = mtgp.predict(X_test, return_std=True)
```

## 📁 Examples

The library includes comprehensive example scripts in the `examples/` directory:

| Example | Description |
|---------|-------------|
| `01_ADM1_simulation.py` | Full ADM1 pipeline with ACoD preprocessing |
| `02_am2_simulation.py` | Basic AM2 model simulation |
| `03_am2_calibration.py` | Parameter calibration with Optuna |
| `04_lstm_prediction.py` | LSTM-based biogas prediction |
| `05_mtgp_prediction.py` | Multi-task GP with uncertainty |
| `06_am2_mpc_control.py` | MPC for biogas maximization |
| `07_am2_vfa_tracking.py` | MPC for VFA setpoint tracking |

Run any example:
```bash
python examples/01_ADM1_simulation.py
```

All plots are automatically saved to the `images/` directory with consistent styling.

## 📂 Package Structure

```
src/openad_lib/
├── feedstock/
│   ├── descriptors.py        # FeedstockDescriptor, CoDigestionMixture
│   └── feedstock_library.py  # Built-in substrate database
├── models/
│   ├── mechanistic/
│   │   ├── adm1_model.py     # ADM1Model (38 states)
│   │   └── am2_model.py      # AM2Model (4 states)
│   └── ml/
│       ├── lstm_model.py     # LSTMModel
│       └── mtgp.py           # MultitaskGP
├── preprocessing/
│   └── acod.py               # ACoD influent characterization
├── optimisation/
│   └── am2_calibrator.py     # Optuna-based calibration
├── control/
│   └── am2_mpc.py            # MPC controller (do-mpc)
├── utils/
│   ├── data_utils.py         # Data loading utilities
│   └── plot_utils.py         # Consistent plotting style
└── data/                     # Sample datasets
```

## 📊 Built-in Feedstocks

The library includes characterization data for common AD substrates:

| Substrate | TS (kg/m³) | VS (g/kg TS) | BMP (NL CH4/kg VS) |
|-----------|------------|--------------|---------------------|
| Maize Silage | 312.8 | 947.2 | 293 |
| Wholecrop | 374.3 | 930.0 | 320 |
| Chicken Litter | 613.3 | 870.4 | 280 |
| Grass | 295.0 | 965.0 | 290 |
| Apple Pomace | 155.8 | 982.7 | 270 |
| ... | ... | ... | ... |

Access via: `FeedstockLibrary().list_feedstocks()`

## 🔬 Model Details

### ADM1 (Anaerobic Digestion Model No. 1)
- **38 State Variables**: 12 soluble, 12 particulate, 6 ion states, 3 gas phase
- **Biochemical Processes**: Disintegration, hydrolysis, uptake, decay
- **Physicochemical Processes**: Acid-base equilibria, gas-liquid transfer
- **Outputs**: Biogas/CH4 production, VFA, pH, biomass dynamics

### AM2 (Two-step Simplified Model)
- **4 State Variables**: S1 (COD), S2 (VFA), X1 (acidogens), X2 (methanogens)
- **Key Processes**: Acidogenesis, methanogenesis with Haldane inhibition
- **Use Cases**: Control design, optimization, real-time applications

## 📈 Model Selection Guide

| Use Case | Recommended Model |
|----------|-------------------|
| Process understanding & optimization | ADM1Model |
| Control design & MPC | AM2Model |
| Real-time prediction | LSTMModel |
| Multi-output with uncertainty | MultitaskGP |
| Limited training data | MultitaskGP |
| Long-term forecasting | ADM1Model |

## 📚 Citation

If you use OpenAD-lib in your research, please cite:

```bibtex
@software{openad_lib,
  author = {Dekhici, Benaissa},
  title = {OpenAD-lib: Open-Source Framework for Uncertainty-Aware Anaerobic Digestion Digital Twins},
  year = {2026},
  url = {https://github.com/benmola/OpenAD-lib}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

- **Author**: Benaissa Dekhici
- **Email**: b.dekhici@surrey.ac.uk
- **Institution**: University of Surrey
