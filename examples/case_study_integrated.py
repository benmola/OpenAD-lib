"""
OpenAD-lib Integrated Case Study: Digital Twin for Biogas Plant Optimization
=============================================================================

This script demonstrates the full potential of OpenAD-lib by creating an integrated
digital twin workflow for a co-digestion biogas plant.

Workflow:
0. Phase 0: Probabilistic Feedstock Characterization (Uncertainty Quantification)
1. Phase 1: ADM1 Mechanistic Baseline
2. Phase 2: AM2 Model Calibration (Optuna)
3. Phase 3A: LSTM Time Series Prediction
4. Phase 3B: Multi-Task GP with Uncertainty
5. Phase 4A: MPC Biogas Maximization
6. Phase 4B: MPC VFA Tracking

Author: OpenAD-lib Team
Date: 2026
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path for development
current_dir = Path(__file__).parent.resolve()
src_path = current_dir.parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import openad_lib as openad

# Create results directory
results_dir = current_dir / 'case_study_results'
results_dir.mkdir(exist_ok=True)

print("=" * 80)
print("OpenAD-lib Integrated Case Study: Digital Twin for Biogas Optimization")
print("=" * 80)

# =============================================================================
# PHASE 0: PROBABILISTIC FEEDSTOCK CHARACTERIZATION
# =============================================================================
print("\n" + "=" * 80)
print("PHASE 0: Probabilistic Feedstock Characterization with Uncertainty")
print("=" * 80)

print("\n[0.1] Initializing feedstock library...")
lib = openad.FeedstockLibrary()
print(f"  ✓ Loaded library with {len(lib.list_feedstocks())} feedstocks")

print("\n[0.2] User provides sparse measurements (realistic scenario)...")
# Realistic: User only has limited lab measurements for maize
user_maize_data = {
    'ts': [310, 315, 308],      # Only 3 TS measurements from lab
    'bmp': [285, 300, 293, 290, 295, 292, 298, 294]           # Only 2 BMP measurements
    # Missing: vs, cod_total, proteins, lipids
}
print(f"  User measurements:")
print(f"    TS:  {user_maize_data['ts']} (n={len(user_maize_data['ts'])})")
print(f"    BMP: {user_maize_data['bmp']} (n={len(user_maize_data['bmp'])})")
print(f"    Missing: VS, COD, proteins, lipids")

print("\n[0.3] Library fills missing parameters with built-in uncertainty...")
maize_prob = lib.get_probabilistic("Maize", user_data=user_maize_data)
print(f"  ✓ Created probabilistic feedstock: {maize_prob.name}")
print(f"    TS:  {maize_prob.ts:.1f} kg/m³ (from USER)")
print(f"    VS:  {maize_prob.vs:.1f} g/kg TS (from LIBRARY)")
print(f"    BMP: {maize_prob.bmp:.1f} NL CH4/kg VS (from USER)")
print(f"    COD: {maize_prob.cod_total:.1f} kg COD/m³ (from LIBRARY)")

print("\n[0.4] Generating ensemble for uncertainty propagation...")
maize_ensemble = maize_prob.sample(n=500, random_state=42)
ts_ensemble = [f.ts for f in maize_ensemble]
vs_ensemble = [f.vs for f in maize_ensemble]
bmp_ensemble = [f.bmp for f in maize_ensemble]
cod_ensemble = [f.cod_total for f in maize_ensemble]
print(f"  ✓ Generated {len(maize_ensemble)} realizations")
print(f"    TS ensemble:  {np.mean(ts_ensemble):.1f} ± {np.std(ts_ensemble):.1f} kg/m³")
print(f"    BMP ensemble: {np.mean(bmp_ensemble):.1f} ± {np.std(bmp_ensemble):.1f} NL CH4/kg VS")
print(f"  → Uncertainty will propagate through ADM1 simulations")

print("\n[0.5] Visualizing sparse user data + library uncertainty...")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Phase 0: Probabilistic Feedstock Characterization', fontsize=20, fontweight='bold')

# TS - User provided
ax = axes[0, 0]
user_ts = np.array(user_maize_data['ts'])
ax.hist(user_ts, bins=3, alpha=0.6, label=f'User Data (n={len(user_ts)})', 
        color='darkgreen', edgecolor='black', width=2)
ax.hist(ts_ensemble, bins=30, alpha=0.5, label=f'Ensemble (n={len(ts_ensemble)})', 
        color='lightblue')
ax.axvline(np.mean(user_ts), color='darkgreen', linestyle='--', linewidth=2, label='Mean (user)')
ax.axvline(np.mean(ts_ensemble), color='blue', linestyle='--', linewidth=1.5, label='Mean (ensemble)')
ax.set_xlabel('Total Solids [kg/m³]', fontweight='bold', fontsize=16)
ax.set_ylabel('Frequency', fontsize=16)
ax.set_title('TS: User Provided', color='darkgreen', fontsize=18)
ax.legend(fontsize=18)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=14)

# VS - Library filled
ax = axes[0, 1]
ax.hist(vs_ensemble, bins=30, alpha=0.7, label=f'Library Ensemble (n={len(vs_ensemble)})', 
        color='orange', edgecolor='black')
ax.axvline(np.mean(vs_ensemble), color='red', linestyle='--', linewidth=2, label='Mean')
ax.text(0.05, 0.95, 'Library Filled\n(No user data)', transform=ax.transAxes, 
        fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.set_xlabel('Volatile Solids [g/kg TS]', fontweight='bold', fontsize=16)
ax.set_ylabel('Frequency', fontsize=16)
ax.set_title('VS: Library Filled', color='orange', fontsize=18)
ax.legend(fontsize=18)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=14)

# BMP - User provided
ax = axes[1, 0]
user_bmp = np.array(user_maize_data['bmp'])
ax.hist(user_bmp, bins=8, alpha=0.6, label=f'User Data (n={len(user_bmp)})', 
        color='darkgreen', edgecolor='black', width=2)
ax.hist(bmp_ensemble, bins=30, alpha=0.5, label=f'Ensemble (n={len(bmp_ensemble)})', 
        color='lightblue')
ax.axvline(np.mean(user_bmp), color='darkgreen', linestyle='--', linewidth=2, label='Mean (user)')
ax.axvline(np.mean(bmp_ensemble), color='blue', linestyle='--', linewidth=1.5, label='Mean (ensemble)')
ax.set_xlabel('BMP [NL CH4/kg VS]', fontweight='bold', fontsize=16)
ax.set_ylabel('Frequency', fontsize=16)
ax.set_title('BMP: User Provided', color='darkgreen', fontsize=18)
ax.legend(fontsize=18)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=14)

# COD - Library filled
ax = axes[1, 1]
ax.hist(cod_ensemble, bins=30, alpha=0.7, label=f'Library Ensemble (n={len(cod_ensemble)})', 
        color='orange', edgecolor='black')
ax.axvline(np.mean(cod_ensemble), color='red', linestyle='--', linewidth=2, label='Mean')
ax.text(0.05, 0.95, 'Library Filled\n(No user data)', transform=ax.transAxes, 
        fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.set_xlabel('Total COD [kg COD/m³]', fontweight='bold', fontsize=16)
ax.set_ylabel('Frequency', fontsize=16)
ax.set_title('COD: Library Filled', color='orange', fontsize=18)
ax.legend(fontsize=18)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=14)

plt.tight_layout()
phase0_plot_path = results_dir / "phase0_probabilistic_feedstock.png"
plt.savefig(phase0_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved plot: {phase0_plot_path}")

# Use mean feedstock for deterministic phases (backward compatibility)
maize_deterministic = lib.get("Maize")
print(f"\n[0.6] Using deterministic feedstock for baseline comparison...")
print(f"  ✓ Maize (deterministic): TS={maize_deterministic.ts:.1f}, BMP={maize_deterministic.bmp:.1f}")

# =============================================================================
# PHASE 1: ADM1 MECHANISTIC BASELINE
# =============================================================================
print("\n" + "=" * 80)
print("PHASE 1: ADM1 Mechanistic Baseline")
print("=" * 80)

print("\n[1.1] Loading feedstock and biogas data...")
feedstock_data = openad.load_sample_data('feedstock')
biogas_data = openad.load_sample_data('biogas')
print(f"  ✓ Loaded {len(feedstock_data)} feedstock samples")
print(f"  ✓ Loaded {len(biogas_data)} biogas measurements")

print("\n[1.2] Generating influent characterization (ACoD)...")
influent_df = openad.acod.generate_influent_data(feedstock_data)
print(f"  ✓ Generated influent data: {influent_df.shape}")

print("\n[1.3] Running ADM1 simulation (38 states)...")
adm1_model = openad.ADM1Model()
adm1_results = adm1_model.simulate(influent_df)
df_qgas = adm1_results['q_gas']
qgas_sim = df_qgas['q_gas'].values
print(f"  ✓ Simulation complete: {len(qgas_sim)} timesteps")

print("\n[1.4] Evaluating ADM1 performance...")
target_col = 'Biogas (m3/day)'
y_true = biogas_data[target_col].values
min_len = min(len(y_true), len(qgas_sim))
y_true_aligned = y_true[:min_len]
y_pred_aligned = qgas_sim[:min_len]
time_aligned = biogas_data['time'].values[:min_len] if 'time' in biogas_data.columns else np.arange(min_len)

adm1_metrics = openad.utils.metrics.compute_metrics(y_true_aligned, y_pred_aligned)
openad.utils.metrics.print_metrics(adm1_metrics, title="ADM1 Performance (Biogas)")

print("\n[1.5] Plotting ADM1 results...")
openad.plots.plot_predictions(
    y_true=y_true_aligned,
    y_pred=y_pred_aligned,
    x=time_aligned,
    title="Phase 1: ADM1 Mechanistic Baseline",
    xlabel="Time (days)",
    ylabel="Biogas Production (m³/day)",
    save_path=str(results_dir / "phase1_adm1_baseline.png"),
    show=False
)
print(f"  ✓ Saved plot: {results_dir / 'phase1_adm1_baseline.png'}")

# =============================================================================
# PHASE 2: AM2 MODEL CALIBRATION (OPTUNA)
# =============================================================================
print("\n" + "=" * 80)
print("PHASE 2: AM2 Model Calibration with Optuna")
print("=" * 80)

print("\n[2.1] Loading AM2 experimental data...")
am2_data = openad.load_sample_data('am2_lab')
print(f"  ✓ Loaded {len(am2_data)} AM2 measurements")

print("\n[2.2] Initializing AM2 model...")
am2_model = openad.AM2Model()
am2_model.load_data_from_dataframe(
    am2_data,
    S1in_col='SCODin',
    S1out_col='SCODout',
    S2out_col='VFAout',
    Q_col='Biogas'
)
print("  ✓ AM2 model initialized with data")

print("\n[2.3] Running initial simulation (before calibration)...")
initial_results = am2_model.run(verbose=False)
initial_metrics = am2_model.evaluate()
openad.utils.metrics.print_metrics(initial_metrics, title="AM2 Initial Performance")

print("\n[2.4] Calibrating parameters with Optuna (30 trials)...")
calibrator = openad.AM2Calibrator(am2_model)
best_params = calibrator.calibrate(
    params_to_tune=['m1', 'K1', 'm2', 'Ki', 'K2'],
    n_trials=30,
    weights={'S1': 0.5, 'S2': 1.0, 'Q': 1.0}
)
print(f"\n  ✓ Calibration complete!")
print(f"  Best parameters: {best_params}")

print("\n[2.5] Running calibrated simulation...")
am2_model.update_params(best_params)
final_results = am2_model.run(verbose=False)
final_metrics = am2_model.evaluate()
openad.utils.metrics.print_metrics(final_metrics, title="AM2 Calibrated Performance")

print("\n[2.6] Plotting calibration comparison...")
openad.plots.plot_calibration_comparison(
    initial_results,
    final_results,
    save_path=str(results_dir / "phase2_am2_calibration.png"),
    show=False
)
print(f"  ✓ Saved plot: {results_dir / 'phase2_am2_calibration.png'}")

# =============================================================================
# PHASE 3A: LSTM TIME SERIES PREDICTION
# =============================================================================
print("\n" + "=" * 80)
print("PHASE 3A: LSTM Time Series Prediction")
print("=" * 80)

print("\n[3A.1] Loading LSTM time series data...")
lstm_data = openad.load_sample_data('lstm_timeseries').dropna()
print(f"  ✓ Loaded {len(lstm_data)} samples")

features = ['Maize', 'Wholecrop', 'Chicken Litter', 'Lactose', 'Apple Pomace', 'Rice bran']
label = 'Total_Biogas'
n_in = 1

print(f"\n[3A.2] Initializing LSTM model...")
input_dim = len(features) * n_in
lstm_model = openad.LSTMModel(input_dim=input_dim, hidden_dim=24, output_dim=1)
print(f"  ✓ LSTM initialized (input_dim={input_dim}, hidden_dim=24)")

print(f"\n[3A.3] Preparing time series data (lags={n_in})...")
X, y, dataset = lstm_model.prepare_time_series_data(
    lstm_data,
    features=features,
    target=label,
    n_in=n_in
)

split_idx = int(len(X) * 0.8)
X_train, y_train = X[:split_idx], y[:split_idx].ravel()
X_test, y_test = X[split_idx:], y[split_idx:].ravel()
print(f"  ✓ Train: {len(X_train)}, Test: {len(X_test)}")

print(f"\n[3A.4] Training LSTM (30 epochs)...")
lstm_model.train(X_train, y_train, epochs=30, verbose=False)
print("  ✓ Training complete")

print(f"\n[3A.5] Evaluating LSTM...")
lstm_metrics = lstm_model.evaluate(X_test, y_test)
openad.utils.metrics.print_metrics(lstm_metrics, title="LSTM Test Performance")

print(f"\n[3A.6] Plotting LSTM predictions...")
train_pred = lstm_model.predict(X_train)
test_pred = lstm_model.predict(X_test)
y_full = np.concatenate([y_train, y_test])
pred_full = np.concatenate([train_pred, test_pred])
train_idx = np.arange(len(y_train))
test_idx = np.arange(len(y_train), len(y_full))

openad.plots.plot_predictions(
    y_true=y_full,
    y_pred=pred_full,
    train_indices=train_idx,
    test_indices=test_idx,
    title="Phase 3A: LSTM Biogas Prediction",
    xlabel="Sample Index",
    ylabel="Biogas Production",
    save_path=str(results_dir / "phase3a_lstm_prediction.png"),
    show=False
)
print(f"  ✓ Saved plot: {results_dir / 'phase3a_lstm_prediction.png'}")

# =============================================================================
# PHASE 3B: MULTI-TASK GP WITH UNCERTAINTY
# =============================================================================
print("\n" + "=" * 80)
print("PHASE 3B: Multi-Task GP with Uncertainty Quantification")
print("=" * 80)

print("\n[3B.1] Loading multi-task GP data...")
mtgp_data = openad.load_sample_data('mtgp')
print(f"  ✓ Loaded {len(mtgp_data)} samples")

input_cols = ['time', 'D', 'SCODin', 'OLR', 'pH']
output_cols = ['SCODout', 'VFAout', 'Biogas']

X = mtgp_data[input_cols].values
Y = mtgp_data[output_cols].values

print(f"\n[3B.2] Splitting data (alternating indices)...")
train_indices = np.arange(1, len(X), 2)
test_indices = np.arange(0, len(X), 2)
X_train, Y_train = X[train_indices], Y[train_indices]
X_test, Y_test = X[test_indices], Y[test_indices]
print(f"  ✓ Train: {len(X_train)}, Test: {len(X_test)}")

print(f"\n[3B.3] Initializing Multi-Task GP (3 tasks, 3 latents)...")
mtgp_model = openad.MultitaskGP(
    num_tasks=len(output_cols),
    num_latents=min(3, len(output_cols)),
    n_inducing=60,
    learning_rate=0.1,
    log_transform=True
)
print("  ✓ MTGP initialized")

print(f"\n[3B.4] Training MTGP (300 iterations)...")
mtgp_model.train(X_train, Y_train, epochs=300, verbose=False)
print("  ✓ Training complete")

print(f"\n[3B.5] Evaluating MTGP...")
mtgp_metrics = mtgp_model.evaluate(X_test, Y_test, task_names=output_cols)
for task_name, task_metrics in mtgp_metrics.items():
    openad.utils.metrics.print_metrics(task_metrics, title=f"MTGP Task: {task_name}")

print(f"\n[3B.6] Plotting MTGP predictions with uncertainty...")
n_train = len(X_train)
train_idx = np.arange(n_train)
test_idx = np.arange(n_train, n_train + len(X_test))
X_full = np.vstack([X_train, X_test])
Y_full = np.vstack([Y_train, Y_test])
mean_full, lower_full, upper_full = mtgp_model.predict(X_full, return_std=True)

openad.plots.plot_multi_output(
    y_true=Y_full,
    y_pred=mean_full,
    x=X_full[:, 0],
    y_lower=lower_full,
    y_upper=upper_full,
    train_indices=train_idx,
    test_indices=test_idx,
    output_names=output_cols,
    title="Phase 3B: Multi-Task GP with Uncertainty",
    xlabel="Time",
    save_path=str(results_dir / "phase3b_mtgp_uncertainty.png"),
    show=False
)
print(f"  ✓ Saved plot: {results_dir / 'phase3b_mtgp_uncertainty.png'}")

# =============================================================================
# PHASE 4A: MPC BIOGAS MAXIMIZATION
# =============================================================================
print("\n" + "=" * 80)
print("PHASE 4A: MPC for Biogas Maximization")
print("=" * 80)

print("\n[4A.1] Initializing MPC controller...")
params = openad.AM2Parameters()
mpc_controller = openad.AM2MPC(params)

sampling_time = 1.0
horizon = 10

mpc_controller.setup_controller(
    sampling_time=sampling_time,
    horizon=horizon,
    objective_type='maximize_biogas',
    D_max=0.5
)
mpc_controller.setup_simulator(sampling_time=sampling_time)
print(f"  ✓ MPC configured (horizon={horizon} days, sampling={sampling_time} day)")

print("\n[4A.2] Setting initial conditions...")
x0 = np.array([5.0, 1.0, 10.0, 1.0])  # [S1, X1, S2, X2]
mpc_controller.set_initial_state(x0)
print(f"  ✓ Initial state: S1={x0[0]}, X1={x0[1]}, S2={x0[2]}, X2={x0[3]}")

print("\n[4A.3] Running MPC simulation (50 days)...")
n_days = 50
S1in_nominal = 15.0
pH_nominal = 7.0

history = {
    'time': [], 'D': [],
    'S1': [], 'X1': [], 'S2': [], 'X2': [], 'Q': []
}

t_current = 0.0
for k in range(n_days):
    # Sinusoidal disturbance in inlet concentration
    S1in_k = S1in_nominal + 5.0 * np.sin(2 * np.pi * k / 20)
    pH_k = pH_nominal
    
    u_opt, y_next = mpc_controller.run_step(S1in_val=S1in_k, pH_val=pH_k)
    
    # Calculate biogas production
    S2_curr = float(y_next[2])
    X2_curr = float(y_next[3])
    mu2_base = params.m2 * (S2_curr / ((S2_curr**2)/params.Ki + S2_curr + params.K2))
    pH_factor = np.exp(-4 * ((pH_k - params.pHH)/(params.pHH - params.pHL))**2)
    mu2 = mu2_base * (1.0 - pH_factor)
    Q_curr = float(params.k6 * mu2 * X2_curr * params.c)
    
    history['time'].append(t_current)
    history['D'].append(u_opt)
    history['S1'].append(y_next[0])
    history['X1'].append(y_next[1])
    history['S2'].append(y_next[2])
    history['X2'].append(y_next[3])
    history['Q'].append(Q_curr)
    
    t_current += sampling_time

print(f"  ✓ MPC simulation complete ({n_days} days)")
print(f"  Average biogas: {np.mean(history['Q']):.2f} m³/day")
print(f"  Average dilution rate: {np.mean(history['D']):.3f} day⁻¹")

print("\n[4A.4] Plotting MPC results...")
openad.plots.plot_mpc_results(
    history,
    d_max=0.5,
    title="Phase 4A: MPC Maximizing Biogas Production",
    save_path=str(results_dir / "phase4a_mpc_biogas_max.png"),
    show=False
)
print(f"  ✓ Saved plot: {results_dir / 'phase4a_mpc_biogas_max.png'}")

# =============================================================================
# PHASE 4B: MPC VFA TRACKING
# =============================================================================
print("\n" + "=" * 80)
print("PHASE 4B: MPC for VFA Setpoint Tracking")
print("=" * 80)

print("\n[4B.1] Initializing VFA tracking controller...")
mpc_tracking = openad.AM2MPC(params)
mpc_tracking.setup_controller(
    sampling_time=sampling_time,
    horizon=horizon,
    objective_type='tracking',
    tracking_variable='S2',
    setpoint=2.0,
    D_max=0.5
)
mpc_tracking.setup_simulator(sampling_time=sampling_time)
mpc_tracking.set_initial_state(x0)
print(f"  ✓ VFA tracking MPC configured (setpoint=2.0 g/L)")

print("\n[4B.2] Running VFA tracking simulation (50 days)...")
tracking_history = {
    'time': [], 'D': [],
    'S1': [], 'X1': [], 'S2': [], 'X2': [], 'Q': [],
    'Setpoint': []
}

t_current = 0.0
for k in range(n_days):
    S1in_k = S1in_nominal + 5.0 * np.sin(2 * np.pi * k / 20)
    pH_k = pH_nominal
    
    u_opt, y_next = mpc_tracking.run_step(S1in_val=S1in_k, pH_val=pH_k)
    
    S2_curr = float(y_next[2])
    X2_curr = float(y_next[3])
    mu2_base = params.m2 * (S2_curr / ((S2_curr**2)/params.Ki + S2_curr + params.K2))
    pH_factor = np.exp(-4 * ((pH_k - params.pHH)/(params.pHH - params.pHL))**2)
    mu2 = mu2_base * (1.0 - pH_factor)
    Q_curr = float(params.k6 * mu2 * X2_curr * params.c)
    
    tracking_history['time'].append(t_current)
    tracking_history['D'].append(u_opt)
    tracking_history['S1'].append(y_next[0])
    tracking_history['X1'].append(y_next[1])
    tracking_history['S2'].append(y_next[2])
    tracking_history['X2'].append(y_next[3])
    tracking_history['Q'].append(Q_curr)
    tracking_history['Setpoint'].append(2.0)
    
    t_current += sampling_time

print(f"  ✓ VFA tracking simulation complete")
print(f"  Average VFA: {np.mean(tracking_history['S2']):.2f} g/L (target: 2.0)")
print(f"  VFA tracking error: {np.abs(np.mean(tracking_history['S2']) - 2.0):.3f} g/L")

print("\n[4B.3] Plotting VFA tracking results...")
openad.plots.plot_mpc_results(
    tracking_history,
    d_max=0.5,
    title="Phase 4B: MPC VFA Setpoint Tracking",
    save_path=str(results_dir / "phase4b_mpc_vfa_tracking.png"),
    show=False
)
print(f"  ✓ Saved plot: {results_dir / 'phase4b_mpc_vfa_tracking.png'}")

# =============================================================================
# SUMMARY REPORT
# =============================================================================
print("\n" + "=" * 80)
print("CASE STUDY SUMMARY")
print("=" * 80)

summary_data = {
    'Phase': [
        '1: ADM1 Baseline',
        '2: AM2 Calibration',
        '3A: LSTM Prediction',
        '3B: MTGP Uncertainty',
        '4A: MPC Biogas Max',
        '4B: MPC VFA Tracking'
    ],
    'Model': ['ADM1', 'AM2', 'LSTM', 'MTGP', 'AM2-MPC', 'AM2-MPC'],
    'R² Score': [
        f"{adm1_metrics.get('R2', 0):.3f}",
        f"{final_metrics.get('R2', 0):.3f}",
        f"{lstm_metrics.get('R2', 0):.3f}",
        f"{mtgp_metrics['Biogas'].get('R2', 0):.3f}",
        'N/A',
        'N/A'
    ],
    'RMSE': [
        f"{adm1_metrics.get('RMSE', 0):.2f}",
        f"{final_metrics.get('RMSE', 0):.2f}",
        f"{lstm_metrics.get('RMSE', 0):.2f}",
        f"{mtgp_metrics['Biogas'].get('RMSE', 0):.2f}",
        'N/A',
        'N/A'
    ],
    'Key Feature': [
        'Mechanistic (38 states)',
        'Optuna calibration',
        'Time series forecasting',
        'Multi-output + uncertainty',
        'Optimal control',
        'Setpoint tracking'
    ]
}

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

# Save summary
summary_file = results_dir / 'case_study_summary.csv'
summary_df.to_csv(summary_file, index=False)
print(f"\n✓ Summary saved: {summary_file}")

print("\n" + "=" * 80)
print("CASE STUDY COMPLETE!")
print("=" * 80)
print(f"\nAll results saved to: {results_dir}")
print("\nGenerated files:")
print("  • phase1_adm1_baseline.png")
print("  • phase2_am2_calibration.png")
print("  • phase3a_lstm_prediction.png")
print("  • phase3b_mtgp_uncertainty.png")
print("  • phase4a_mpc_biogas_max.png")
print("  • phase4b_mpc_vfa_tracking.png")
print("  • case_study_summary.csv")
print("\nThis case study demonstrates OpenAD-lib's integrated capabilities:")
print("  ✓ Mechanistic modeling (ADM1)")
print("  ✓ Parameter calibration (Optuna)")
print("  ✓ ML surrogates (LSTM, MTGP)")
print("  ✓ Uncertainty quantification")
print("  ✓ Model predictive control")
print("=" * 80)
