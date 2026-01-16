"""
Example: Using the New Simplified OpenAD-lib API
================================================

This script demonstrates the improved package structure with:
1. Simplified imports at package level
2. Configuration management
3. Data loading utilities
4. Dataset classes

All Priority 1 improvements implemented!
"""

import sys
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent.resolve()
src_path = current_dir.parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# ============================================================================
# 1. SIMPLIFIED IMPORTS - No more deep nesting!
# ============================================================================
print("="*70)
print("PRIORITY 1 IMPROVEMENTS DEMO")
print("="*70)

print("\n1️⃣  SIMPLIFIED PACKAGE-LEVEL IMPORTS")
print("-" * 70)

# Old way (still works for backward compatibility):
# from openad_lib.models.mechanistic import ADM1Model
# from openad_lib.models.ml import LSTMModel

# New way - much cleaner!
import openad_lib as oad

print("✓ Imported openad_lib as oad")
print(f"✓ Package version: {oad.__version__}")

# Access models directly
ADM1Model = oad.ADM1Model
AM2Model = oad.AM2Model
LSTMModel = oad.LSTMModel
print("✓ Accessed ADM1Model, AM2Model, LSTMModel from package level")

# ============================================================================
# 2. CONFIGURATION MANAGEMENT
# ============================================================================
print("\n\n2️⃣  CONFIGURATION MANAGEMENT")
print("-" * 70)

# Access global config
print(f"Default device: {oad.config.default_device}")
print(f"ODE solver: {oad.config.ode_solver}")
print(f"Verbose mode: {oad.config.verbose}")
print(f"Data directory: {oad.config.data_dir}")

# Modify configuration
oad.config.verbose = False
oad.config.ode_solver = 'RK45'
print("\n✓ Modified config: verbose=False, ode_solver='RK45'")

# View all config settings
print("\nFull configuration:")
for key, value in oad.config.to_dict().items():
    print(f"  {key}: {value}")

# ============================================================================
# 3. DATA LOADING UTILITIES
# ============================================================================
print("\n\n3️⃣  DATA LOADING UTILITIES")
print("-" * 70)

# Load sample data easily
try:
    biogas_data = oad.load_sample_data('biogas')
    print(f"✓ Loaded biogas sample data: {biogas_data.shape}")
    print(f"  Columns: {biogas_data.columns.tolist()}")
    
    feedstock_data = oad.load_sample_data('feedstock')
    print(f"✓ Loaded feedstock sample data: {feedstock_data.shape}")
    print(f"  Columns: {feedstock_data.columns.tolist()}")
    
except Exception as e:
    print(f"Note: Sample data loading: {e}")

# ============================================================================
# 4. DATASET CLASSES
# ============================================================================
print("\n\n4️⃣  DATASET CLASSES")
print("-" * 70)

# Create dataset objects
try:
    # BiogasDataset
    biogas_dataset = oad.BiogasDataset(
        biogas_data,
        time_column='time',
        biogas_column='Biogas'
    )
    print(f"✓ Created BiogasDataset: {biogas_dataset}")
    
    # Split into train/test
    train_ds, test_ds = biogas_dataset.split(train_fraction=0.8)
    print(f"  Train set: {len(train_ds)} samples")
    print(f"  Test set: {len(test_ds)} samples")
    
    # FeedstockDataset
    feedstock_dataset = oad.FeedstockDataset(
        feedstock_data,
        time_column='time'
    )
    print(f"✓ Created FeedstockDataset: {feedstock_dataset}")
    print(f"  Feedstocks: {feedstock_dataset.feedstocks}")
    
except Exception as e:
    print(f"Note: Dataset creation: {e}")

# ============================================================================
# 5. USING MODELS WITH NEW API
# ============================================================================
print("\n\n5️⃣  USING MODELS WITH SIMPLIFIED API")
print("-" * 70)

# Initialize models using simplified imports
print("Initializing models...")
adm1 = oad.ADM1Model()
print(f"✓ ADM1Model: {adm1.__class__.__name__}")

am2 = oad.AM2Model()
print(f"✓ AM2Model: {am2.__class__.__name__}")

# ML models
lstm = oad.LSTMModel(input_dim=6, hidden_dim=24, output_dim=1)
print(f"✓ LSTMModel: {lstm.__class__.__name__}")

# ============================================================================
# 6. COMPARISON: OLD vs NEW API
# ============================================================================
print("\n\n6️⃣  API COMPARISON")
print("-" * 70)

print("OLD WAY (still works):")
print("  from openad_lib.models.mechanistic import ADM1Model")
print("  from openad_lib.models.ml import LSTMModel")
print("  from openad_lib.optimisation import ADM1Calibrator")

print("\nNEW WAY (cleaner!):")
print("  import openad_lib as oad")
print("  model = oad.ADM1Model()")
print("  lstm = oad.LSTMModel(...)")
print("  calibrator = oad.ADM1Calibrator(...)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n\n" + "="*70)
print("PRIORITY 1 IMPROVEMENTS SUMMARY")
print("="*70)

improvements = [
    "✓ Package-level API simplification (import openad_lib as oad)",
    "✓ Configuration management (oad.config)",
    "✓ Data loading utilities (load_sample_data, load_csv_data)",
    "✓ Data validation functions (validate_influent_data, etc.)",
    "✓ Dataset classes (BiogasDataset, FeedstockDataset, TimeSeriesDataset)",
    "✓ Backward compatibility maintained",
    "✓ Lazy imports for optional dependencies"
]

for improvement in improvements:
    print(improvement)

print("\n" + "="*70)
print("All Priority 1 improvements successfully implemented!")
print("="*70)
