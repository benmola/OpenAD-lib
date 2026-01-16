# OpenAD-lib

**Open-Source Framework for Uncertainty-Aware Anaerobic Digestion Digital Twins**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

OpenAD-lib is a unified Python library for anaerobic digestion (AD) process modelling, control, and scheduling with explicit uncertainty quantification. It provides both mechanistic models (ADM1, AM2) and machine learning surrogates (LSTM, Multi-Task GP) for biogas production prediction.

## 🚀 Features

- **Mechanistic Models**: Complete ADM1 (38 state variables) and simplified AM2 (4 state variables)
- **ML Surrogate Models**: LSTM and Multi-Task Gaussian Process with uncertainty quantification
- **Unified Plotting System**: Standardized, publication-ready visualization across all modules
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

# Install all dependencies (including ML, optimization, control)
pip install -e .
```

### Using conda

```bash
# Create a new environment
conda create -n openad python=3.10
conda activate openad

# Install dependencies
pip install -e .
```

## 🏃 Quick Start

> **New in v0.2.0**: Simplified API! Import everything from the top level.

```python
import openad_lib as openad

# Access models, calibrators, and utilities directly
model = openad.AM2Model()
calibrator = openad.ADM1Calibrator(model, data, influent)
dataset = oad.BiogasDataset.from_csv('biogas_data.csv')
```

### 1. ADM1 Simulation with ACoD Preprocessing

```python
import openad_lib as oad

# Generate influent data from feedstock ratios
influent_df = oad.acod.generate_influent_data("path/to/feed_ratios.csv")

# Initialize and run ADM1 simulation
model = oad.ADM1Model()
results = model.simulate(influent_df)

# Access results
biogas = results['q_gas']
states = results['results']

# Evaluate against measurements
measured_data = oad.load_sample_data('biogas')
metrics = model.evaluate(measured_data['Biogas'].values, biogas['q_gas'].values)
oad.utils.metrics.print_metrics(metrics)
```

### 2. AM2 Simplified Model

```python
import openad_lib as oad

# Initialize model with default parameters
model = oad.AM2Model()

# Load experimental data (using built-in loader for example)
data = oad.load_sample_data('am2_lab')
model.load_data_from_dataframe(data)

# Run simulation
results = model.simulate()

# Evaluate
metrics = model.evaluate()
oad.utils.metrics.print_metrics(metrics)

# Plot using unified system
oad.plots.plot_multi_output(
    y_true=model.data[['SCODout', 'VFAout', 'Biogas']].values,
    y_pred=results[['S1', 'S2', 'Q']].values,
    output_names=['SCOD', 'VFA', 'Biogas'],
    save_plot=True,
    show=True
)
```

### 3. AM2 Parameter Calibration

```python
import openad_lib as oad

# Initialize model and load data
model = oad.AM2Model()
data = oad.load_sample_data('am2_lab')
model.load_data_from_dataframe(data)

# Configure calibrator
calibrator = oad.AM2Calibrator(model)

# Run optimization
best_params = calibrator.calibrate(
    params_to_tune=['m1', 'K1', 'm2', 'Ki', 'K2'],
    n_trials=50,
    weights={'S1': 0.5, 'S2': 1.0, 'Q': 1.0}
)

# Visualize Comparison
initial_results = model.simulate()
model.update_parameters(best_params)
final_results = model.simulate()

oad.plots.plot_calibration_comparison(
    initial_results,
    final_results,
    save_plot=True,
    show=True
)
```

### 4. Configuration Management

```python
import openad_lib as oad

# View current configuration
print(oad.config.default_device)  # 'cpu'
print(oad.config.ode_solver)      # 'LSODA'

# Modify settings
oad.config.default_device = 'cuda'
oad.config.verbose = False
oad.config.ode_solver = 'RK45'

# View all settings
config_dict = oad.config.to_dict()
```

### 5. Data Loading and Validation

```python
import openad_lib as oad
from openad_lib.data import validate_influent_data, validate_feedstock_data

# Load sample datasets
biogas_data = oad.load_sample_data('biogas')
feedstock_data = oad.load_sample_data('feedstock')

# Create dataset objects
biogas_ds = oad.BiogasDataset(biogas_data)
train_ds, test_ds = biogas_ds.split(train_fraction=0.8)

# Validate data
results = validate_influent_data(influent_df, model_type='adm1')
if all(results.values()):
    print("Data is valid!")
```

### 6. LSTM Surrogate Model

```python
import openad_lib as oad

# Load and prepare data
data = oad.load_sample_data('lstm_timeseries')
features = ['Maize', 'Wholecrop', 'Chicken Litter', 'Lactose', 'Apple Pomace', 'Rice bran']
target = 'Total_Biogas'

# Create LSTM model
lstm = oad.LSTMModel(input_dim=len(features), hidden_dim=24, output_dim=1)

# Prepare time series data (built-in helper)
X, y, dataset = lstm.prepare_time_series_data(data, features, target, n_in=1)

# Split and train
split_idx = int(len(X) * 0.8)
X_train, y_train = X[:split_idx], y[:split_idx]
X_test, y_test = X[split_idx:], y[split_idx:]

lstm.train(X_train, y_train, epochs=50, batch_size=4)

# Evaluate
metrics = lstm.evaluate(X_test, y_test)
print(f"Test RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.3f}")
```

### 7. Multi-Task GP with Uncertainty

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

## 📓 Interactive Notebooks (Google Colab)

Learn by running these tutorials directly in your browser:

| Notebook | Description | Open in Colab |
|----------|-------------|---------------|
| [01_ADM1_Tutorial](notebooks/01_ADM1_Tutorial_Updated.ipynb) | Full ADM1 mechanistic model | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/benmola/OpenAD-lib/blob/main/notebooks/01_ADM1_Tutorial_Updated.ipynb) |
| [02_AM2_Modelling](notebooks/02_AM2_Modelling_Updated.ipynb) | AM2 simulation & calibration | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/benmola/OpenAD-lib/blob/main/notebooks/02_AM2_Modelling_Updated.ipynb) |
| [03_LSTM_Prediction](notebooks/03_LSTM_Prediction_Updated.ipynb) | LSTM surrogate model | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/benmola/OpenAD-lib/blob/main/notebooks/03_LSTM_Prediction_Updated.ipynb) |
| [04_MTGP_Prediction](notebooks/04_MTGP_Prediction_Updated.ipynb) | Multi-Task GP with uncertainty | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/benmola/OpenAD-lib/blob/main/notebooks/04_MTGP_Prediction_Updated.ipynb) |
| [05_MPC_Control](notebooks/05_MPC_Control_Updated.ipynb) | Model Predictive Control | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/benmola/OpenAD-lib/blob/main/notebooks/05_MPC_Control_Updated.ipynb) |

## 📚 References

### Core Publications

| Component | Reference |
|-----------|-----------|
| **LSTM for AD** | [Murali et al. (2025) - LAPSE](https://psecommunity.org/LAPSE:2025.0213) |
| **Multi-Task GP** | [Dekhici et al. (2025) - LAPSE](https://psecommunity.org/LAPSE:2025.0155) |
| **AM2 Modelling & Calibration** | [Dekhici et al. (2024) - ACM DL](https://dl.acm.org/doi/10.1145/3680281) |
| **Data-Driven Control** | [Dekhici et al. (2024) - ResearchGate](https://www.researchgate.net/publication/378298857_Data-Driven_Modeling_Order_Reduction_and_Control_of_Anaerobic_Digestion_Processes) |
| **ACoD & Feedstock Library** | [Astals et al. (2015) - PubMed](https://pubmed.ncbi.nlm.nih.gov/27088248/) |
| **ADM1 Implementation** | [PyADM1 - GitHub](https://github.com/CaptainFerMag/PyADM1), [Rosén & Jeppsson (2021)](https://www.biorxiv.org/content/biorxiv/early/2021/03/04/2021.03.03.433746.full.pdf) |
| **Optuna (Calibration)** | [Akiba et al. (2019) - KDD Paper](https://arxiv.org/abs/1907.10902), [GitHub](https://github.com/optuna/optuna) |
| **do-mpc Framework** | [do-mpc Documentation](https://www.do-mpc.com) |

## 📝 Citation

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

