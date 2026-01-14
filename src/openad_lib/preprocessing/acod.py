
import pandas as pd
import numpy as np
import os

def calculate_values(df_substrate, df_conversion_factors, B_0_values, VFA_values, M_N, ratios):
    """
    Calculate weighted substrate characteristics based on mixing ratios.
    """
    weighted_substrate_data = {}
    
    # Calculate weighted averages based on ratios
    for column in df_substrate.columns[1:]: # Skip 'Substrate' name column
        if df_substrate[column].dtype.kind in 'biufc': # Only numeric columns
            weighted_sum = 0
            for substrate, ratio in ratios.items():
                if substrate in df_substrate['Substrate'].values:
                    val = df_substrate.loc[df_substrate['Substrate'] == substrate, column].values[0]
                    weighted_sum += val * ratio
            weighted_substrate_data[column] = weighted_sum

    weighted_B_0 = sum(B_0_values.get(key, 0) * ratio for key, ratio in ratios.items())
    weighted_VFA = sum(VFA_values.get(key, 0) * ratio for key, ratio in ratios.items())

    df_weighted_substrate = pd.DataFrame([weighted_substrate_data])

    # ADM1 Input Calculation Logic
    # ----------------------------
    
    # f_d: Fraction of biodegradability
    # Formula: (B0 / (350 * CODt)) * VS
    # Note: 350 is approx theoretical CH4 yield per g COD removed (0.35 L/g COD). 
    # But units here are kg/m3. 350 nmL/gVS is typical BMP. 
    # Lets follow the user's logic: (B0 / (350 * CODt)) * VS. Wait, CODt is kg/m3. 
    # User formula: f_d = (weighted_B_0 / (350 * df_weighted_substrate['CODt [kg COD.m-3]'])) * df_weighted_substrate['VS [kg.m-3]']
    
    # Safety wrapper for division by zero if necessary, though unlikely given inputs
    f_d = (weighted_B_0 / (350 * df_weighted_substrate['CODt [kg COD.m-3]'])) * df_weighted_substrate['VS [kg.m-3]']
    
    S_I = df_weighted_substrate['CODs [kg COD.m-3]'] * (1 - f_d)
    X_I = df_weighted_substrate['CODp [kg COD.m-3]'] * (1 - f_d)

    conversion_factors_dict = df_conversion_factors.set_index('Parameter').to_dict()['Value']

    C_li = df_weighted_substrate['Lipids [kg.m-3]']
    C_pr = df_weighted_substrate['Proteins [kg.m-3]']
    
    X_li = C_li * conversion_factors_dict['Yli'] * f_d
    X_pr = C_pr * conversion_factors_dict['Ypr'] * f_d
    X_ch = (df_weighted_substrate['CODp [kg COD.m-3]'] * f_d) - X_pr - X_li

    C_ac = df_weighted_substrate['Acetate [kg.m-3]']
    C_pro = df_weighted_substrate['Propionate [kg.m-3]']
    C_bu = df_weighted_substrate['Butyrate [kg.m-3]']
    C_va = df_weighted_substrate['Valerate [kg.m-3]']
    
    S_ac = C_ac * conversion_factors_dict['Yac']
    S_pro = C_pro * conversion_factors_dict['Ypro']
    S_bu = C_bu * conversion_factors_dict['Ybu']
    S_va = C_va * conversion_factors_dict['Yva']

    VFA_t = weighted_VFA
    
    # Distribute remaining COD to soluble substrates
    # Logic from user code: ((CODs * fd) - VFAt) * fraction
    # Note: user uses 'X_ch'/'CODp' ratio for S_su allocation? Check user code carefully.
    # S_su = ((df_weighted_substrate['CODs [kg COD.m-3]'] * f_d) - VFA_t) * (X_ch / (df_weighted_substrate['CODp [kg COD.m-3]'] * f_d))
    
    cod_avail = (df_weighted_substrate['CODs [kg COD.m-3]'] * f_d) - VFA_t
    cod_p_deg = df_weighted_substrate['CODp [kg COD.m-3]'] * f_d
    
    S_su = cod_avail * (X_ch / cod_p_deg)
    S_aa = cod_avail * (X_pr / cod_p_deg)
    S_fa = cod_avail * (X_li / cod_p_deg)

    S_IN = df_weighted_substrate['TAN [g N.m-3]'] / (M_N * 1000)

    results = pd.DataFrame({
        'f_d': f_d,
        'S_I': S_I,
        'X_I': X_I,
        'X_li': X_li,
        'X_pr': X_pr,
        'X_ch': X_ch,
        'S_ac': S_ac,
        'S_pro': S_pro,
        'S_bu': S_bu,
        'S_va': S_va,
        'S_su': S_su,
        'S_aa': S_aa,
        'S_fa': S_fa,
        'S_IN': S_IN
    })

    return results

def generate_influent_data(ratios_csv_path, output_csv_path=None, substrate_data=None):
    """
    Main function to generate influent data for ADM1 simulation.
    
    Args:
        ratios_csv_path: Path to the CSV containing substrate ratios over time.
        output_csv_path: Optional path to save the generated influent parameters. If None, no file is saved.
        substrate_data: Optional dictionary overriding default substrate properties.
    """
    
    # Default Substrate Data (from user code)
    if substrate_data is None:
        substrate_data_dict = {
             'Substrate':            ['Maize',  'Wholecrop', 'Chicken Litter', 'Lactose', 'Apple Pomace', 'Rice bran', 'Crimped Wheat', 'Grass', 'Corn Gluten/Rapemeal', 'Potatoes', 'FYM', 'Fodder Beet'],
             'TS [kg.m-3]':          [312.8,        374.3, 613.3, 141.9, 155.8, 733.0, 750.0, 295.0, 877.0, 159.4, 294.7, 618.0],
             'VS [kg.m-3]':          [947.2,  930.0, 870.4, 798.1, 982.7, 920.0, 300.0, 965.0, 845.9, 936.1, 956.0, 370.0],
             'CODt [kg COD.m-3]':    [421.7,   400.0, 350.0, 300.0, 320.0, 380.0, 410.0, 370.0, 390.0, 340.0, 360.0, 330.0],
             'CODs [kg COD.m-3]':    [52.7,   60.0, 50.0, 40.0, 45.0, 55.0, 65.0, 50.0, 60.0, 45.0, 50.0, 40.0],
             'CODp [kg COD.m-3]':    [369, 340.0, 300.0, 260.0, 275.0, 325.0, 345.0, 320.0, 330.0, 295.0, 310.0, 290.0],
             'Acetate [kg.m-3]':     [1.84,  2.0, 1.5, 1.0, 1.2, 1.8, 2.2, 1.7, 2.0, 1.5, 1.8, 1.6],
             'Propionate [kg.m-3]':  [0.0,  0.5, 0.3, 0.2, 0.3, 0.4, 0.6, 0.5, 0.6, 0.4, 0.5, 0.4],
             'Butyrate [kg.m-3]':    [0.0,   0.2, 0.1, 0.05, 0.1, 0.15, 0.25, 0.2, 0.25, 0.15, 0.2, 0.18],
             'Valerate [kg.m-3]':    [0.01,   0.1, 0.05, 0.03, 0.05, 0.08, 0.12, 0.1, 0.12, 0.08, 0.1, 0.09],
             'Proteins [kg.m-3]':    [31.1,   30, 25, 20, 22, 28, 32, 27, 30, 24, 26, 23],
             'Lipids [kg.m-3]':      [7.7,  5.0, 3.0, 2.0, 2.5, 4.0, 6.0, 4.5, 5.5, 3.5, 4.0, 3.8],
             'TAN [g N.m-3]':        [457,  300, 200, 150, 180, 250, 350, 270, 320, 230, 260, 240]
        }
        df_substrate = pd.DataFrame(substrate_data_dict)
    else:
        df_substrate = substrate_data

    # Constants and Lookups
    B_0_values = {'Maize': 293,  'Wholecrop': 320, 'Chicken Litter': 280, 'Lactose': 250, 'Apple Pomace': 270, 'Rice bran': 300, 'Crimped Wheat': 310, 'Grass': 290, 'Corn Gluten/Rapemeal': 300, 'Potatoes': 260, 'FYM': 280, 'Fodder Beet': 270}
    VFA_values = {'Maize': 1.84,  'Wholecrop': 2.0, 'Chicken Litter': 1.5, 'Lactose': 1.0, 'Apple Pomace': 1.2, 'Rice bran': 1.8, 'Crimped Wheat': 2.2, 'Grass': 1.7, 'Corn Gluten/Rapemeal': 2.0, 'Potatoes': 1.5, 'FYM': 1.8, 'Fodder Beet': 1.6}
    
    conversion_factors = {
        'Parameter': ['Yac', 'Ypro', 'Ybu', 'Yva', 'Ypr', 'Yli'],
        'Value': [1.06667, 1.513514, 1.818182, 2.039216, 1.53, 2.878]
    }
    df_conversion_factors = pd.DataFrame(conversion_factors)
    M_N = 14.007

    # Load Data
    print(f"Reading ratios from {ratios_csv_path}")
    ratios_df = pd.read_csv(ratios_csv_path)
    
    # Identify substrate columns (exclude known metadata columns if any, or just use intersection)
    # The user's Data.csv has 'time' and many other cols. We only want substrates.
    # Substrates list:
    substrates = df_substrate['Substrate'].values.tolist()
    
    # Create result dataframe
    calculated_parameters_df = ratios_df.copy()

    # Pre-allocate columns to ensure float dtype preventing FutureWarning
    # We do a dummy calculation or just set them to NaN float
    dummy_calcs = calculate_values(df_substrate, df_conversion_factors, B_0_values, VFA_values, M_N,  {sub: 0 for sub in substrates})
    for col in dummy_calcs.columns:
        if col not in calculated_parameters_df.columns:
            calculated_parameters_df[col] = 0.0
            calculated_parameters_df[col] = calculated_parameters_df[col].astype(float)
        else:
             calculated_parameters_df[col] = calculated_parameters_df[col].astype(float)
    
    print("Calculating influent parameters...")
    for index, row in ratios_df.iterrows():
        # Build ratio map for this timestep
        current_ratios = {}
        total_ratio = 0
        
        for sub in substrates:
            if sub in row:
                val = row[sub]
                if pd.notna(val) and isinstance(val, (int, float)):
                    current_ratios[sub] = val
                    total_ratio += val
        
        if total_ratio > 0:
            # Normalize ratios
            normalized_ratios = {k: v / total_ratio for k, v in current_ratios.items()}
            
            # Calculate ADM1 variables
            calcs = calculate_values(df_substrate, df_conversion_factors, B_0_values, VFA_values, M_N, normalized_ratios)
            
            # Append/Update the row
            for col in calcs.columns:
                calculated_parameters_df.at[index, col] = calcs.iloc[0][col]
        else:
            # Handle zero ratio (maintain previous or set to 0?)
            # Assuming 0 contribution
            pass

    # Define exact requested columns
    requested_columns = [
        "time", "f_d", "S_su", "S_aa", "S_fa", "S_va", "S_bu", "S_pro", "S_ac", "S_h2", "S_ch4", "S_IC", "S_IN", "S_I",
        "X_xc", "X_ch", "X_pr", "X_li", "X_su", "X_aa", "X_fa", "X_c4", "X_pro", "X_ac", "X_h2", "X_I",
        "S_cation", "S_anion", "q_ad", "T",
        "S_H_ion", "S_va_ion", "S_bu_ion", "S_pro_ion", "S_ac_ion", "S_hco3_ion", "S_co2", "S_nh3", "S_nh4_ion",
        "S_gas_h2", "S_gas_ch4", "S_gas_co2"
    ]
    
    # Ensure all requested columns exist
    for col in requested_columns:
        if col not in calculated_parameters_df.columns:
             calculated_parameters_df[col] = 0.0
             
    # Select only the requested columns in the specific order
    # Note: If any required column (like 'time') is missing from input, this might raise KeyError if not handled.
    # 'time', 'q_ad', 'T' are expected to be in ratios_df/calculated_parameters_df from the input CSV.
    
    final_df = calculated_parameters_df[requested_columns]

    if output_csv_path:
        print(f"Saving generated input data to {output_csv_path}")
        # User's code saves to 'ADM1_Influent_.csv'
        final_df.to_csv(output_csv_path, index=False)
        
    return final_df
