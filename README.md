# OpenAD-lib

**Open-Source Framework for Uncertainty-Aware Anaerobic Digestion Digital Twins**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

OpenAD-lib is a unified Python library for anaerobic digestion (AD) process modelling, control, and scheduling with explicit uncertainty quantification. It provides both mechanistic models (ADM1) and machine learning surrogates (LSTM, Multi-Task GP) for biogas production prediction.

## 🚀 Features

- **Mechanistic Models**: Complete ADM1 implementation (38 state variables, COD-based)
- **ML Surrogate Models**: LSTM and Multi-Task Gaussian Process with uncertainty quantification
- **Feedstock Library**: Built-in database of 12 common AD substrates
- **Uncertainty Propagation**: From feedstock variability through model predictions
- **Easy Integration**: Simple API for both mechanistic and data-driven approaches

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

### 1. Feedstock Characterization

```python
from openad_lib.feedstock import FeedstockLibrary

# Load the built-in feedstock library
lib = FeedstockLibrary()

# Get a specific feedstock
maize = lib.get("Maize")
print(f"Maize BMP: {maize.bmp} NL CH4/kg VS")

# Create a co-digestion mixture
mixture = lib.create_mixture(
    feedstock_names=["Maize", "Chicken Litter"],
    ratios=[0.7, 0.3]
)
print(f"Mixture BMP: {mixture.bmp:.1f} NL CH4/kg VS")

# Calculate ADM1 inputs from feedstock
adm1_inputs = lib.calculate_adm1_inputs(maize, flow_rate=130)
```

### 2. ADM1 Mechanistic Model

```python
from openad_lib.models.mechanistic import ADM1Model

# Initialize the model
model = ADM1Model()

# Configure reactor parameters
model.params.V_liq = 10000  # m³
model.params.V_gas = 600    # m³
model.params.T_op = 318.15  # K (35°C)

# Load data
model.load_data(
    influent_path="path/to/influent.xlsx",
    initial_state_path="path/to/initial.csv",
    influent_sheet="Influent_ADM1_COD_Based"
)

# Run simulation
results = model.run(solver_method="BDF", verbose=True)

# Plot results
model.plot_results()

# Get results as DataFrames
states_df, gas_flow_df = model.get_results()
```

### 3. LSTM Surrogate Model

```python
from openad_lib.models.ml import LSTMModel
import pandas as pd

# Load your data
data = pd.read_csv("feedstock_timeseries.csv")
X = data[['Maize', 'Grass', 'Manure']].values
y = data['Biogas'].values

# Create and train model
lstm = LSTMModel(input_dim=3, hidden_dim=24, output_dim=1)
lstm.fit(X_train, y_train, epochs=50, batch_size=4)

# Make predictions
predictions = lstm.predict(X_test)

# Evaluate
metrics = lstm.evaluate(X_test, y_test)
print(f"Test RMSE: {metrics['rmse']:.2f}")
print(f"Test R²: {metrics['r2']:.3f}")

# Cross-validate with time series splits
cv_results = lstm.cross_validate(X, y, n_splits=5)
```

### 4. Multi-Task GP with Uncertainty

```python
from openad_lib.models.ml import MultitaskGP

# Predict multiple outputs simultaneously (SCOD, VFA, Biogas)
mtgp = MultitaskGP(num_tasks=3, num_latents=3)

# Train
mtgp.fit(X_train, Y_train, epochs=100)

# Predict with uncertainty
mean, lower, upper = mtgp.predict(X_test, return_std=True)

# The 95% confidence interval shows prediction uncertainty
print(f"Biogas prediction: {mean[:, 2]} ± {(upper[:, 2] - lower[:, 2])/2}")
```

## 📂 Package Structure

```
src/openad_lib/
├── feedstock/
│   ├── descriptors.py       # FeedstockDescriptor, CoDigestionMixture
│   └── feedstock_library.py # Built-in substrate database
├── models/
│   ├── mechanistic/
│   │   └── adm1_model.py    # ADM1Model (38 states)
│   └── ml/
│       ├── lstm_model.py    # LSTMModel
│       └── mtgp.py          # MultitaskGP
├── utils/
│   ├── data_utils.py        # Data loading utilities
│   └── visualisation.py     # Plotting functions
└── data/                    # Sample datasets
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

## 🔬 ADM1 Model Details

The ADM1 implementation follows the IWA Anaerobic Digestion Model No. 1 with BSM2 parameters:

- **38 State Variables**: 12 soluble, 12 particulate, 6 ion states, 3 gas phase
- **Biochemical Processes**: Disintegration, hydrolysis, uptake, decay
- **Physicochemical Processes**: Acid-base equilibria, gas-liquid transfer
- **Solver**: Hybrid ODE-DAE with Newton-Raphson for algebraic constraints

Key outputs:
- Biogas/methane production (m³/day)
- VFA concentrations (acetate, propionate, butyrate)
- pH, FOS/TAC ratios
- Biomass dynamics

## 📈 Model Selection Guide

| Use Case | Recommended Model |
|----------|-------------------|
| Process understanding & optimization | ADM1Model |
| Real-time prediction | LSTMModel |
| Multi-output with uncertainty | MultitaskGP |
| Limited training data | MultitaskGP |
| Long-term forecasting | ADM1Model |

## 🧪 Running Examples

```bash
cd examples
python example_usage.py
```

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
