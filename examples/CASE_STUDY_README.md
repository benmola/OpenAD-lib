# OpenAD-lib Case Study: Integrated Digital Twin

## Overview

This case study demonstrates the **full potential of OpenAD-lib** by creating an integrated digital twin for a co-digestion biogas plant. It showcases how multiple library components work together to solve real-world optimization challenges.

## Scenario

**Agricultural Biogas Plant (500 kW)** processing multiple feedstocks (maize, grass, chicken litter) with challenges:
- Fluctuating feedstock quality
- Need to maximize biogas while maintaining stability
- VFA accumulation risk
- Limited historical data

## What This Case Study Demonstrates

### ✅ Complete Workflow Integration
- Probabilistic characterization → Mechanistic modeling → Calibration → ML surrogates → Optimal control
- 5 phases covering all major OpenAD-lib features
- ~30-45 minutes total runtime

### ✅ Key Features Showcased

| Phase | Component | Feature Demonstrated |
|-------|-----------|---------------------|
| **0** | Feedstock Library | Probabilistic characterization + sparse data |
| **1** | ADM1 | Mechanistic baseline (32 states) |
| **2** | AM2 + Optuna | Parameter calibration |
| **3A** | LSTM | Time series prediction |
| **3B** | MTGP | Multi-output + uncertainty |
| **4A** | MPC | Biogas maximization |
| **4B** | MPC | VFA setpoint tracking |

## Quick Start

### Run the Complete Case Study

```bash
cd examples
python case_study_integrated.py
```

**Expected runtime**: ~30-45 minutes

### Output Files

All results saved to `examples/case_study_results/`:
- `phase0_probabilistic_feedstock.png` - Sparse user data + library uncertainty
- `phase1_adm1_baseline.png` - ADM1 validation
- `phase2_am2_calibration.png` - Before/after calibration
- `phase3a_lstm_prediction.png` - LSTM forecasting
- `phase3b_mtgp_uncertainty.png` - Multi-output with confidence intervals
- `phase4a_mpc_biogas_max.png` - Optimal control for biogas
- `phase4b_mpc_vfa_tracking.png` - VFA tracking control
- `case_study_summary.csv` - Performance metrics table

## Detailed Workflow

### Phase 0: Probabilistic Feedstock Characterization
**Goal**: Handle sparse user measurements with library uncertainty filling

```python
# Initialize feedstock library
lib = openad.FeedstockLibrary()

# User provides LIMITED measurements (realistic scenario)
user_maize_data = {
    'ts': [310, 315, 308],    # Only 3 TS measurements from lab
    'bmp': [285, 300, 293, 290, 295, 292, 298, 294]  # 8 BMP measurements
    # Missing: vs, cod_total, proteins, lipids
}

# Library fills missing parameters automatically
maize_prob = lib.get_probabilistic("Maize", user_data=user_maize_data)

# Generate ensemble for uncertainty propagation
maize_ensemble = maize_prob.sample(n=500, random_state=42)
```

**Key Features**:
- **Sparse Data Handling**: User provides only available measurements
- **Automatic Gap Filling**: Library fills missing parameters from built-in uncertainty
- **Source Tracking**: Clear labels showing USER vs LIBRARY data
- **Ensemble Generation**: 100 realizations for uncertainty propagation

**Visualization**: 4-panel plot showing:
- Top-Left: TS (user provided - 3 measurements)
- Top-Right: VS (library filled - no user data)
- Bottom-Left: BMP (user provided - 8 measurements)
- Bottom-Right: COD (library filled - no user data)

---

### Phase 1: ADM1 Mechanistic Baseline
**Goal**: Establish physics-based understanding

```python
# Generate influent from feedstock ratios
influent_df = openad.acod.generate_influent_data(feedstock_data)

# Run ADM1 simulation (32 states)
adm1_model = openad.ADM1Model()
results = adm1_model.simulate(influent_df)

# Evaluate performance
metrics = openad.utils.metrics.compute_metrics(y_true, y_pred)
```

**Outputs**: Biogas predictions, VFA dynamics, pH trajectory

---

### Phase 2: AM2 Calibration with Optuna
**Goal**: Tune simplified model to plant data

```python
# Initialize AM2 model
am2_model = openad.AM2Model()
am2_model.load_data_from_dataframe(data, ...)

# Calibrate with Optuna (30 trials)
calibrator = openad.AM2Calibrator(am2_model)
best_params = calibrator.calibrate(
    params_to_tune=['m1', 'K1', 'm2', 'Ki', 'K2'],
    n_trials=30
)
```

**Outputs**: Optimized parameters, before/after comparison

---

### Phase 3A: LSTM Time Series Prediction
**Goal**: Fast surrogate for real-time prediction

```python
# Create LSTM model
lstm = openad.LSTMModel(input_dim=6, hidden_dim=24)

# Prepare time series with lags
X, y, dataset = lstm.prepare_time_series_data(data, features, target, n_in=1)

# Train and evaluate
lstm.train(X_train, y_train, epochs=30)
metrics = lstm.evaluate(X_test, y_test)
```

**Outputs**: Predictions with train/test split visualization

---

### Phase 3B: Multi-Task GP with Uncertainty
**Goal**: Multi-output prediction with confidence intervals

```python
# Initialize MTGP (3 outputs: SCOD, VFA, Biogas)
mtgp = openad.MultitaskGP(num_tasks=3, num_latents=3)

# Train and predict with uncertainty
mtgp.train(X_train, Y_train, epochs=300)
mean, lower, upper = mtgp.predict(X_test, return_std=True)
```

**Outputs**: Simultaneous predictions for 3 outputs with 95% CI

---

### Phase 4A: MPC Biogas Maximization
**Goal**: Optimize feeding strategy for maximum biogas

```python
# Setup MPC controller
mpc = openad.AM2MPC(params)
mpc.setup_controller(
    horizon=10,
    objective_type='maximize_biogas',
    D_max=0.5
)

# Run closed-loop simulation
for k in range(30):
    u_opt, y_next = mpc.run_step(S1in_val=S1in_k, pH_val=pH_k)
```

**Outputs**: Optimal dilution rate trajectory, biogas production

---

### Phase 4B: MPC VFA Tracking
**Goal**: Maintain VFA at safe setpoint

```python
# Configure VFA tracking
mpc.setup_controller(
    objective_type='track_vfa',
    vfa_setpoint=2.0
)

# Run tracking simulation
for k in range(30):
    u_opt, y_next = mpc.run_step(...)
```

**Outputs**: VFA tracking performance, control effort

---

## Key Advantages Demonstrated

### 🎯 Unified Interface
```python
import openad_lib as openad  # Single import
model = openad.ADM1Model()   # Consistent API
data = openad.load_sample_data('biogas')  # Built-in data
```

### 🔬 Hybrid Modeling
- **ADM1**: Detailed mechanistic understanding (38 states)
- **AM2**: Fast model for control design (4 states)
- **LSTM/MTGP**: Data-driven surrogates for real-time use

### 📊 Uncertainty Quantification
- Feedstock variability propagation
- MTGP epistemic uncertainty
- Confidence intervals for decision-making

### 🎨 Publication-Ready Outputs
- Standardized plotting across all modules
- Automatic saving to `images/` directory
- Consistent styling and formatting

### ⚡ End-to-End Workflow
- Data → Preprocessing → Modeling → Calibration → Control
- All in one script, fully reproducible

---

## Expected Performance

| Metric | Target | Typical Result |
|--------|--------|----------------|
| ADM1 R² (Biogas) | > 0.75 | 0.78-0.85 |
| AM2 R² (calibrated) | > 0.85 | 0.87-0.92 |
| LSTM RMSE | < 15% mean | 10-12% |
| MTGP Coverage | > 90% | 92-95% |
| MPC Improvement | > 10% | 12-18% |

---

## Customization Options

### Different Feedstocks
Modify feedstock ratios in input data to test different substrate combinations.

### Alternative Calibration
```python
# Calibrate different parameters
calibrator.calibrate(
    params_to_tune=['m1', 'm2', 'K1', 'K2', 'Ki', 'pHH', 'pHL'],
    n_trials=100
)
```

### Extended Simulations
```python
# Longer MPC horizon
mpc.setup_controller(horizon=20, ...)

# More training epochs
lstm.train(X_train, y_train, epochs=100)
```

### Economic Optimization
Add electricity price signals to MPC objective for revenue optimization.

---

## Use Cases

### 📚 Research & Publications
- Demonstrates complete digital twin methodology
- Publication-ready figures and metrics
- Reproducible workflow for peer review

### 🏭 Industrial Demonstrations
- Shows practical applicability to real plants
- Quantifies performance improvements
- Validates control strategies

### 🎓 Educational Tutorials
- Step-by-step workflow explanation
- Covers all major AD modeling techniques
- Hands-on learning with real data

### 💰 Grant Proposals
- Showcases technical capabilities
- Demonstrates innovation potential
- Provides concrete performance metrics

---

## Next Steps

### Extend the Case Study
1. **Hybrid Models**: Use MTGP to correct ADM1 predictions
2. **Economic MPC**: Include electricity prices
3. **Fault Detection**: Use residuals for anomaly detection
4. **Multi-Plant**: Scale to multiple digesters

### Adapt to Your Data
1. Replace sample data with your plant measurements
2. Adjust feedstock library to your substrates
3. Tune calibration parameters to your process
4. Configure MPC for your operational constraints

---

## References

For detailed documentation on each component:
- **ADM1**: See `examples/01_adm1_simulation.py`
- **AM2 Calibration**: See `examples/03_am2_calibration.py`
- **LSTM**: See `examples/04_lstm_prediction.py`
- **MTGP**: See `examples/05_mtgp_prediction.py`
- **MPC**: See `examples/06_am2_mpc_control.py` and `07_am2_vfa_tracking.py`

---

## Support

For questions or issues:
- **Email**: b.dekhici@surrey.ac.uk
- **GitHub**: [OpenAD-lib Issues](https://github.com/benmola/OpenAD-lib/issues)
- **Documentation**: See main [README.md](../README.md)

---

**This case study demonstrates OpenAD-lib as a complete framework for uncertainty-aware anaerobic digestion digital twins!** 🚀
