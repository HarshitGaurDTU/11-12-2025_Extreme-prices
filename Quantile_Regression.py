import pandas as pd
import numpy as np
import os
import statsmodels.formula.api as smf
import warnings
import re # Import the regular expression module for robust column cleaning

# --- GLOBAL CONFIGURATION ---

# Define input and output paths based on user request
INPUT_FILE_PATH = r'C:\Users\harsh\OneDrive\Documents\Master thesis\Raw data\Finalized 2024 data\Dataset_germany_fossil_fuels_with_2024.csv'
BASE_RESULTS_DIR = r'C:\Users\harsh\OneDrive\Documents\Master thesis\Raw data\Finalized 2024 data\Results\Part 2'
REGRESSION_OUTPUT_PATH = os.path.join(BASE_RESULTS_DIR, "Quantile_Regression_Coefficients_By_Year.xlsx")

# Define relevant column names
DEPENDENT_VAR = 'Price'
YEAR_COL = 'Year'

# New list of independent variables, excluding time factors (Day, Hour, Month)
INDEPENDENT_VARS = [
    'Temperature', 
    'Open cycle - Gas Marginal cost',
    'Residual Load',

]

# Combined list for initial data loading
COLS_TO_USE = [DEPENDENT_VAR, YEAR_COL] + INDEPENDENT_VARS

# --- 1. DATA PREPARATION (Simplified to just load and rename) ---

def sanitize_column_name(name):
    """Replaces all non-alphanumeric/underscore characters with an underscore."""
    # Replace any sequence of non-alphanumeric/non-underscore characters with a single underscore
    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # Remove leading/trailing underscores and potential double underscores
    safe_name = re.sub(r'_{2,}', '_', safe_name).strip('_')
    return safe_name

def load_and_rename_data(file_path):
    """Loads, cleans, and renames data using a robust sanitization function."""
    
    # Load data
    try:
        df_raw = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Input file not found at {file_path}")

    # --- FIX: Use robust sanitization for all columns ---
    df_raw.columns = [sanitize_column_name(col) for col in df_raw.columns]
    
    # Update COLS_TO_USE list with safe names
    safe_cols = [sanitize_column_name(col) for col in COLS_TO_USE]
    
    # Filter for required columns and drop NaNs
    df_clean = df_raw[safe_cols].copy().dropna()

    if df_clean.empty:
        raise ValueError("DataFrame is empty after cleaning. Check data quality.")
        
    print(f"Data cleaned. Total rows used: {len(df_clean)}")
    
    return df_clean

def standardize_predictors(df):
    """Standardizes predictors within a given DataFrame (e.g., a single year's data)."""
    
    # Use the sanitized names for the independent variables
    safe_independent_vars = [sanitize_column_name(col) for col in INDEPENDENT_VARS]
    df_standardized = df.copy()
    
    for col_safe in safe_independent_vars:
        if col_safe in df_standardized.columns and df_standardized[col_safe].dtype in [np.number, np.float64, np.int64]:
            mean_val = df_standardized[col_safe].mean()
            std_val = df_standardized[col_safe].std()
            if std_val > 0:
                # Create standardized column (e.g., 'Residual_Load_Z')
                df_standardized[f'{col_safe}_Z'] = (df_standardized[col_safe] - mean_val) / std_val
            else:
                # Handle zero std dev (constant column)
                df_standardized[f'{col_safe}_Z'] = 0 
    return df_standardized

# --- 2. QUANTILE REGRESSION EXECUTION (Per Year) ---

def run_yearly_quantile_regression(df_year, year, quantiles):
    """
    Runs Quantile Regression for a single year and specified quantiles.
    """
    
    # Generate the regression formula using only standardized predictors
    safe_independent_vars = [sanitize_column_name(col) for col in INDEPENDENT_VARS]
    formula_vars_z = [f'{col}_Z' for col in safe_independent_vars]
    
    # Check if all Z-score columns exist before creating formula
    missing_z_cols = [z_col for z_col in formula_vars_z if z_col not in df_year.columns]
    if missing_z_cols:
        print(f"   -> Skipping {year}: Missing standardized columns in data: {missing_z_cols}. Check sanitization.")
        return pd.DataFrame()

    formula = f"{DEPENDENT_VAR} ~ {' + '.join(formula_vars_z)}"
    
    all_results = []
    
    for tau in quantiles:
        tau_name = f'Q{int(tau * 100)}'
        
        try:
            # 1. Fit the QR Model
            model = smf.quantreg(formula, data=df_year)
            # Increased max_iter for robustness, which is common for QR
            res = model.fit(q=tau, max_iter=2000) 
            
            # 2. Extract Results
            coefs = res.params.rename('Coefficient')
            pvalues = res.pvalues.rename('P_Value')
            std_err = res.bse.rename('Std_Error')
            
            # Combine into a DataFrame
            # FIX: Capture index (Predictor Name) and rename the index column before concatenation
            df_res = pd.DataFrame({
                'Coefficient': coefs,
                'P_Value': pvalues,
                'Std_Error': std_err
            }).reset_index().rename(columns={'index': 'Predictor'})
            
            df_res['Year'] = year
            df_res['Quantile'] = tau_name
            df_res['Tau'] = tau
            df_res['N_Obs'] = res.nobs
            # FIX: Changed to the correct attribute 'prsquared'
            df_res['Pseudo_R_squared'] = res.prsquared 

            all_results.append(df_res)
            
            # --- Specific check for 'Fossil forecast' coefficient significance ---
            fossil_coef_name = sanitize_column_name('Fossil forecast') + '_Z'
            if fossil_coef_name in df_res['Predictor'].values:
                # Find the row corresponding to the fossil forecast predictor
                fossil_row = df_res[df_res['Predictor'] == fossil_coef_name].iloc[0]
                p_val = fossil_row['P_Value']
                coef = fossil_row['Coefficient']
                sig_level = '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                print(f"   {year} - {tau_name}: {fossil_coef_name} Coef={coef:.4f} {sig_level} (p={p_val:.4f})")
            
        except Exception as e:
            print(f"   -> ERROR: Quantile Regression failed for {year}, tau={tau}. Error: 'QuantRegResults' object has no attribute 'prsquared' (or other issue): {e}")
            
    return pd.concat(all_results) if all_results else pd.DataFrame()


# --- 3. MAIN EXECUTION ---

def main_analysis():
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # 1. Load and Prepare Data
    print(f"Loading data from: {INPUT_FILE_PATH}")
    try:
        df_full = load_and_rename_data(INPUT_FILE_PATH)
    except Exception as e:
        print(f"Error during data loading: {e}")
        return

    # Determine years to iterate over AND enforce the 2019-2024 range
    if sanitize_column_name(YEAR_COL) not in df_full.columns:
        print("Error: 'Year' column not found after renaming. Check data structure.")
        return
        
    all_years = sorted(df_full[sanitize_column_name(YEAR_COL)].unique().astype(int))
    
    # Filter years to run only from 2019 to 2024
    years_to_run = [y for y in all_years if 2019 <= y <= 2024]
    QUANTILES_TO_RUN = [0.10, 0.50, 0.90]

    all_yearly_results = []
    print("\n--- Starting Quantile Regression Analysis (Yearly) ---")
    
    # 2. Loop through each year
    for year in years_to_run:
        # Use sanitized column name for filtering
        df_year = df_full[df_full[sanitize_column_name(YEAR_COL)] == year].copy()
        
        if len(df_year) < 100: # Simple check for minimum data points
            print(f"Skipping year {year}: Too few observations ({len(df_year)}).")
            continue
            
        print(f"\n--- Processing Year: {year} (N={len(df_year)}) ---")
        
        # Standardize the predictors using only the current year's data
        df_year_standardized = standardize_predictors(df_year)
        
        # Run QR for the current year
        yearly_results = run_yearly_quantile_regression(df_year_standardized, year, QUANTILES_TO_RUN)
        
        if not yearly_results.empty:
            all_yearly_results.append(yearly_results)

    # 3. Consolidate and Export Results
    if all_yearly_results:
        # The inner loop now ensures the resulting DataFrames are flat and ready for concatenation
        df_final_results = pd.concat(all_yearly_results)
        
        # Final formatting and column ordering
        df_final_results = df_final_results[['Year', 'Quantile', 'Tau', 'N_Obs', 'Pseudo_R_squared', 'Predictor', 'Coefficient', 'P_Value', 'Std_Error']]
        
        # Ensure output directory exists and save to Excel
        os.makedirs(BASE_RESULTS_DIR, exist_ok=True)
        df_final_results.to_excel(REGRESSION_OUTPUT_PATH, index=False)
        print(f"\nSuccessfully exported all yearly QR results to: {REGRESSION_OUTPUT_PATH}")
        
    else:
        print("\nNo successful yearly regression results were generated.")


if __name__ == '__main__':
    main_analysis()