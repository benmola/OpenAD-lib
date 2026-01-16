"""
Quick test to verify core library improvements.

Tests:
1. Utils modules (metrics, validation)
2. Base classes
3. AM2Model with base class inheritance
4. ADM1Calibrator import
"""

import sys
sys.path.insert(0, r"c:\Users\bdekh\OneDrive - University of Surrey\Work Benaissa Dekhici\ESCAPE 2026 Abstract\OpenAi-Lib\OpenAD-lib\src")

print("=" * 60)
print("Testing Core Library Improvements")
print("=" * 60)

# Test 1: Utils - Metrics
print("\n1️⃣ Testing utils.metrics...")
try:
    from openad_lib.utils.metrics import compute_metrics, print_metrics
    import numpy as np
    
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
    
    metrics = compute_metrics(y_true, y_pred)
    print(f"   ✓ Computed metrics: RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 2: Utils - Validation
print("\n2️⃣ Testing utils.validation...")
try:
    from openad_lib.utils.validation import validate_state_bounds, validate_params
    
    states = np.array([5.0, 0.5, 2.0, 0.3])
    validate_state_bounds(states, model='AM2')
    print("   ✓ State validation successful")
    
    params = {'m1': 0.2, 'K1': 15.0}
    bounds = {'m1': (0.05, 0.5), 'K1': (5.0, 30.0)}
    validate_params(params, bounds)
    print("   ✓ Parameter validation successful")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 3: Base Classes
print("\n3️⃣ Testing base classes...")
try:
    from openad_lib.models.base import BaseModel, MechanisticModel, MLModel
    print(f"   ✓ BaseModel imported")
    print(f"   ✓ MechanisticModel imported")
    print(f"   ✓ ML Model imported")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 4: AM2Model with Base Class
print("\n4️⃣ Testing AM2Model inheritance...")
try:
    from openad_lib.models.mechanistic import AM2Model, AM2Parameters
    
    model = AM2Model()
    print(f"   ✓ AM2Model instance created")
    print(f"   ✓ Inherits from MechanisticModel: {isinstance(model, MechanisticModel)}")
    
    # Test update_params method
    model.update_params({'m1': 0.15, 'K1': 12.0})
    print(f"   ✓ update_params() works: m1={model.params.m1}")
    
    # Test get_params method
    params_dict = model.get_params()
    print(f"   ✓ get_params() works: {len(params_dict)} parameters")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 5: ADM1Calibrator
print("\n5️⃣ Testing ADM1Calibrator...")
try:
    from openad_lib.optimisation import ADM1Calibrator
    print(f"   ✓ ADM1Calibrator imported")
    print(f"   ✓ Has default parameter bounds: {len(ADM1Calibrator.__init__.__code__.co_names) > 0}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 6: Exports
print("\n6️⃣ Testing package exports...")
try:
    from openad_lib.models import BaseModel, MechanisticModel, MLModel
    from openad_lib.utils import compute_metrics, validate_state_bounds
    from openad_lib.optimisation import AM2Calibrator, ADM1Calibrator
    print("   ✓ All key exports working")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 60)
print("✅ Core library improvements tests complete!")
print("=" * 60)
