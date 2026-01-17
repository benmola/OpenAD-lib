import openad_lib as openad
import pandas as pd

# Initialize model and load data
model = openad.AM2Model()
data = openad.load_sample_data('am2_lab')
model.load_data_from_dataframe(
    data,
    S1in_col='SCODin',
    S1out_col='SCODout',
    S2out_col='VFAout',
    Q_col='Biogas'
)

# Test what run() returns
print("Testing model.run()...")
initial_results = model.run(verbose=False)
print(f"Type of initial_results: {type(initial_results)}")
print(f"Is DataFrame: {isinstance(initial_results, pd.DataFrame)}")

if isinstance(initial_results, pd.DataFrame):
    print(f"DataFrame columns: {initial_results.columns.tolist()}")
    print(f"DataFrame shape: {initial_results.shape}")
    
    # Test if we can extract columns
    try:
        test_data = initial_results[['S1', 'S2', 'Q']].values
        print(f"✅ Successfully extracted columns: {test_data.shape}")
    except Exception as e:
        print(f"❌ Error extracting columns: {e}")
else:
    print(f"❌ run() returned {type(initial_results)} instead of DataFrame")
    print(f"Keys: {initial_results.keys() if hasattr(initial_results, 'keys') else 'N/A'}")
