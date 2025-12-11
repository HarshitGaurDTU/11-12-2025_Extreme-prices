import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
import numpy as np

# Suppress warnings that might be irrelevant for simple plotting
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- GLOBAL CONFIGURATION (Based on your context) ---
INPUT_FILE_PATH = r'C:\Users\harsh\OneDrive\Documents\Master thesis\Raw data\Finalized 2024 data\Dataset_germany_fossil_fuels_with_2024.csv'
BASE_RESULTS_DIR = r'C:\Users\harsh\OneDrive\Documents\Master thesis\Raw data\Finalized 2024 data\Results\Part 2\Model\Plots for residual load quad regression'

# Define key column names
PRICE_COL = 'Price'
ORIGINAL_RL_COL = 'Residual Load'
YEAR_COL = 'Year'
SAFE_RL_COL = 'Residual_Load'
RL_COL_TO_USE = SAFE_RL_COL # This is the original RL column name

# Define standardized column names
PRICE_Z_COL = 'Price_Z'
RL_Z_COL = 'RL_Z'

# --- 1. AUXILIARY FUNCTION: QUADRATIC REGRESSION ---

def run_quadratic_regression(df, price_col, rl_col, regime_name):
    """
    Performs quadratic regression (Price_Z ~ RL_Z + RL_Z^2) and calculates R-squared.
    Note: price_col and rl_col are expected to be standardized (Z-scores).
    """
    # Need at least 3 unique X points for a quadratic fit
    if len(df) < 3 or df[rl_col].nunique() < 3: 
        return {
            'Regime': regime_name,
            'Intercept_B0': np.nan,
            'RL_B1': np.nan,
            'RL2_B2': np.nan,
            'R-squared': np.nan,
            'N_Obs': len(df),
            'fit_params': (np.nan, np.nan, np.nan)
        }

    X = df[rl_col]
    Y = df[price_col]

    try:
        # Quadratic fit: Y = B0 + B1*X + B2*X^2
        # Note: B0 here is the intercept when Price and RL are standardized.
        coeffs = np.polyfit(X, Y, 2) 
        
        # Calculate R-squared (R-squared is invariant to standardization)
        p = np.poly1d(coeffs)
        y_hat = p(X)
        y_mean = np.mean(Y)
        ss_total = np.sum((Y - y_mean)**2)
        ss_residual = np.sum((Y - y_hat)**2)
        
        if ss_total == 0:
            r_squared = 1.0 
        else:
            r_squared = 1 - (ss_residual / ss_total)
        
        return {
            'Regime': regime_name,
            'Intercept_B0': coeffs[2], # Intercept of Z-score fit
            'RL_B1': coeffs[1],       # Linear coefficient of Z-score fit
            'RL2_B2': coeffs[0],      # Quadratic coefficient of Z-score fit
            'R-squared': r_squared,
            'N_Obs': len(df),
            'fit_params': (coeffs[2], coeffs[1], coeffs[0]) # (B0, B1, B2)
        }
    except Exception as e:
        print(f"Warning: Regression failed for {regime_name} with error: {e}")
        return {
            'Regime': regime_name,
            'Intercept_B0': np.nan,
            'RL_B1': np.nan,
            'RL2_B2': np.nan,
            'R-squared': np.nan,
            'N_Obs': len(df),
            'fit_params': (np.nan, np.nan, np.nan)
        }

# --- 2. AUXILIARY FUNCTION: PLOTTING FIT LINE (Requires original data for x-axis) ---

def plot_fit_line(ax, df_regime, results, label, color, linestyle='-', mean_rl=0, std_rl=1, mean_price=0, std_price=1):
    """
    Plots the quadratic fit line. It calculates points using standardized 
    coefficients but transforms them back to original units for plotting.
    """
    if results['N_Obs'] >= 3 and not np.isnan(results['R-squared']):
        
        # 1. Define range in original RL units
        min_rl = df_regime[RL_COL_TO_USE].min()
        max_rl = df_regime[RL_COL_TO_USE].max()
        
        # Create 100 points for a smooth fit line
        X_fit_original = np.linspace(min_rl * 0.999, max_rl * 1.001, 100)
        
        # 2. Standardize the x-axis points for prediction
        X_fit_z = (X_fit_original - mean_rl) / std_rl
        
        # 3. Predict the Y values (Price) in Z-score
        B0_z, B1_z, B2_z = results['fit_params']
        Y_fit_z = B0_z + B1_z * X_fit_z + B2_z * X_fit_z**2
        
        # 4. De-standardize the predicted Y values back to original Price units
        Y_fit_original = Y_fit_z * std_price + mean_price
        
        ax.plot(X_fit_original, Y_fit_original, 
                color=color, 
                linewidth=2, 
                linestyle=linestyle,
                label=f'{label} Fit (R²={results["R-squared"]:.2f})')
    
# --- 3. MAIN FUNCTION: MULTI-CASE REGRESSION AND PLOT ---

def run_multi_case_regression_and_plot():
    
    YEARS_TO_ANALYZE = range(2019, 2025) 
    
    PERCENTILE_CASES = [
        (0.99, 0.01),  # P99 / P01
        (0.95, 0.05),  # P95 / P05
        (0.90, 0.10)   # P90 / P10
    ]
    
    # --- OUTPUT SETUP ---
    os.makedirs(BASE_RESULTS_DIR, exist_ok=True)
    print(f"Base output directory confirmed/created at: {BASE_RESULTS_DIR}")

    # --- LOAD AND PREPARE DATA ---
    print(f"\nLoading data from: {INPUT_FILE_PATH}...")
    try:
        df = pd.read_csv(INPUT_FILE_PATH) 

        if ORIGINAL_RL_COL in df.columns:
            df.rename(columns={ORIGINAL_RL_COL: SAFE_RL_COL}, inplace=True)
        
        required_cols = [PRICE_COL, RL_COL_TO_USE, YEAR_COL] 
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}.")

        df_process = df[required_cols].copy() 
        df_process[YEAR_COL] = df_process[YEAR_COL].astype(int) 
        df_process[PRICE_COL] = pd.to_numeric(df_process[PRICE_COL], errors='coerce')
        df_process[RL_COL_TO_USE] = pd.to_numeric(df_process[RL_COL_TO_USE], errors='coerce')
        df_process.dropna(inplace=True, subset=[PRICE_COL, RL_COL_TO_USE, YEAR_COL])
        
        if df_process.empty:
            raise ValueError("DataFrame is empty after cleaning.")

        print(f"Data loaded successfully. Total rows: {len(df_process)}")
    except Exception as e:
        print(f"\nERROR during data loading/preparation: {e}")
        return

    # --- RUN ANALYSIS FOR EACH PERCENTILE CASE ---
    
    all_regression_results = []
    
    for p_upper, p_lower in PERCENTILE_CASES:
        
        p_upper_name = f'P{int(p_upper * 100)}'
        p_lower_name = f'P{int(p_lower * 100)}'
        case_name = f'{p_upper_name}_{p_lower_name}'
        
        print(f"\n=======================================================")
        print(f"| Starting Case: {case_name} (Upper: {p_upper}, Lower: {p_lower})")
        print(f"=======================================================")

        OUTPUT_DIR_PLOTS = os.path.join(BASE_RESULTS_DIR, f"Plots_RL_Price_Hourly_{case_name}_Piecewise_Standardized")
        REGRESSION_OUTPUT_PATH = os.path.join(BASE_RESULTS_DIR, f"RL_Price_Regression_Coefficients_{case_name}_Standardized.xlsx")
        
        os.makedirs(OUTPUT_DIR_PLOTS, exist_ok=True)
        regression_results_case = []

        for year in YEARS_TO_ANALYZE:
            print(f"--- Processing year: {year} ({case_name}) ---")
            
            # Filter data for the current year
            df_year = df_process[df_process[YEAR_COL] == year].copy()

            if df_year.empty:
                print(f"Skipping year {year}: No data available.")
                continue
            
            # --- Perform YEARLY Standardization ---
            mean_rl = df_year[RL_COL_TO_USE].mean()
            std_rl = df_year[RL_COL_TO_USE].std()
            
            mean_price = df_year[PRICE_COL].mean()
            std_price = df_year[PRICE_COL].std()
            
            # Create standardized columns for the current year's data
            df_year[RL_Z_COL] = (df_year[RL_COL_TO_USE] - mean_rl) / std_rl
            df_year[PRICE_Z_COL] = (df_year[PRICE_COL] - mean_price) / std_price

            # --- Calculate Percentiles using the ORIGINAL (non-standardized) RL ---
            rl_lower = df_year[RL_COL_TO_USE].quantile(p_lower)
            rl_upper = df_year[RL_COL_TO_USE].quantile(p_upper)
            
            # Get the standardized RL boundaries for filtering
            # Standardizing the percentiles here ensures consistency with the Z-score RL data
            rl_lower_z = (rl_lower - mean_rl) / std_rl
            rl_upper_z = (rl_upper - mean_rl) / std_rl

            # --- 1. Regime Filtering using Standardized RL (RL_Z_COL) ---
            
            # Normal Regime 
            df_normal = df_year[
                (df_year[RL_Z_COL] >= rl_lower_z) & 
                (df_year[RL_Z_COL] <= rl_upper_z)
            ].copy()
            # Run regression using Z-score columns
            results_normal = run_quadratic_regression(df_normal, PRICE_Z_COL, RL_Z_COL, f'{p_lower_name}_to_{p_upper_name}_Normal')
            results_normal['Year'] = year
            results_normal['Case'] = case_name
            regression_results_case.append(results_normal)

            # Scarcity Regime 
            df_scarcity = df_year[df_year[RL_Z_COL] > rl_upper_z].copy()
            results_scarcity = run_quadratic_regression(df_scarcity, PRICE_Z_COL, RL_Z_COL, f'Above_{p_upper_name}_Scarcity')
            results_scarcity['Year'] = year
            results_scarcity['Case'] = case_name
            regression_results_case.append(results_scarcity)

            # Oversupply Regime 
            df_oversupply = df_year[df_year[RL_Z_COL] < rl_lower_z].copy()
            results_oversupply = run_quadratic_regression(df_oversupply, PRICE_Z_COL, RL_Z_COL, f'Below_{p_lower_name}_Oversupply')
            results_oversupply['Year'] = year
            results_oversupply['Case'] = case_name
            regression_results_case.append(results_oversupply)

            # --- 2. GENERATE PIECEWISE PLOT ---
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Scatter plot of raw data (using original units)
            ax.scatter(df_year[RL_COL_TO_USE], df_year[PRICE_COL], 
                       alpha=0.2, s=15, color='#1f77b4', 
                       label=f'Hourly Obs. {year}')
            
            # Plot the fit lines for the three regimes (de-standardized)
            # Pass the yearly standardization values for plotting transformation
            plot_fit_line(ax, df_normal, results_normal, f'{p_lower_name}-{p_upper_name} Normal', 'orange', linestyle='-', mean_rl=mean_rl, std_rl=std_rl, mean_price=mean_price, std_price=std_price)
            plot_fit_line(ax, df_scarcity, results_scarcity, f'Above {p_upper_name} Scarcity', 'red', linestyle='-', mean_rl=mean_rl, std_rl=std_rl, mean_price=mean_price, std_price=std_price)
            plot_fit_line(ax, df_oversupply, results_oversupply, f'Below {p_lower_name} Oversupply', 'green', linestyle='-', mean_rl=mean_rl, std_rl=std_rl, mean_price=mean_price, std_price=std_price)

            # Add Percentile Lines (using original units)
            ax.axvline(rl_upper, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f'{p_upper_name} RL ({rl_upper:,.0f} GWh)')
            ax.axvline(rl_lower, color='green', linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f'{p_lower_name} RL ({rl_lower:,.0f} GWh)')
            
            # Formatting and Saving
            ax.set_title(f'Piecewise Quadratic Fit (Z-Coefficients): RL vs. Price ({year}) - Case {p_lower_name}-{p_upper_name}', fontsize=16)
            ax.set_xlabel('Residual Load (GWh)', fontsize=12)
            ax.set_ylabel('Day-Ahead Price (€/MWh)', fontsize=12)
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, linestyle=':', alpha=0.6)
            
            # Set y-limit to avoid extreme outliers distorting the curve visualization (e.g., max price of 1000)
            ax.set_ylim(-300, min(df_year[PRICE_COL].max() * 1.05, 1000))
            
            plot_file_name = f"RL_Price_Scatter_Piecewise_Standardized_{case_name}_{year}.png"
            plot_path = os.path.join(OUTPUT_DIR_PLOTS, plot_file_name)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # print(f"Plot saved to: {plot_path}")

        # --- EXPORT REGRESSION RESULTS ---
        df_results_case = pd.DataFrame(regression_results_case)
        # Ensure the 'Case' column is present before selecting columns
        df_results_case = df_results_case[['Year', 'Case', 'Regime', 'N_Obs', 'R-squared', 'Intercept_B0', 'RL_B1', 'RL2_B2']]
        df_results_case.to_excel(REGRESSION_OUTPUT_PATH, index=False)
        print(f"\nCase {case_name} standardized coefficients saved successfully to: {REGRESSION_OUTPUT_PATH}")
        
        all_regression_results.append(df_results_case)
        
    print("\n\nAll percentile cases successfully processed and exported with Z-score coefficients.")

if __name__ == '__main__':
    run_multi_case_regression_and_plot()