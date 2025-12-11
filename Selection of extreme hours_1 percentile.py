import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def analyze_and_plot_extreme_prices(file_path):
    """
    Loads electricity price data, identifies 1st and 99th percentile hours 
    for each year, outputs the raw data for these hours, and generates 
    a plot highlighting them against both Hour and Month of Year.
    
    The results (plot and Excel file) are saved to the specified
    'Part 2' folder.

    Args:
        file_path (str): The local path to the CSV file.
    """
    
    # --- Define and Create Output Folder ---
    output_folder = r'C:\Users\harsh\OneDrive\Documents\Master thesis\Raw data\Finalized 2024 data\Results\Part 2'
    
    try:
        os.makedirs(output_folder, exist_ok=True)
        print(f"Output directory ensured at: {output_folder}")
    except Exception as e:
        print(f"Error creating output directory: {e}")
        return
        
    # --- 1. Data Loading and Validation ---
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at path: {file_path}")
        print("Please ensure the path is correct and accessible.")
        return

    try:
        # Load the raw data
        df = pd.read_csv(file_path)
        
        # Standardize column names for easy access
        required_cols = {'Price', 'Year', 'Month', 'Hour', 'Day'}
        missing_cols = required_cols - set(df.columns)
        
        if missing_cols:
            print(f"Error: The following required columns are missing in the file: {missing_cols}")
            print("Please check the column names in your CSV file.")
            return

        # Ensure correct data types
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df['Year'] = df['Year'].astype(int)
        df['Month'] = df['Month'].astype(int)
        df['Day'] = df['Day'].astype(int)
        df['Hour'] = df['Hour'].astype(int)

        df.dropna(subset=['Price'], inplace=True)
        
        # Combine date columns to create a datetime index (important for time series analysis)
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
        # Adjust Hour to be zero-indexed (0-23) if it's 1-24
        if df['Hour'].max() > 23:
             df['Hour'] = df['Hour'] - 1
             
        df['DateTime'] = df['Date'] + pd.to_timedelta(df['Hour'], unit='h')
        df.set_index('DateTime', inplace=True)
        
    except Exception as e:
        print(f"An error occurred during data loading/processing: {e}")
        return

    # --- 2. Initialize Output and Plotting Setup ---
    all_extreme_hours = []
    years = sorted(df['Year'].unique())
    
    # Setup plotting grid
    num_years = len(years)
    cols = 2
    rows_per_year = 2 # Hour plot and Month plot
    total_rows = int(np.ceil(num_years * rows_per_year / cols))
    
    fig, axes = plt.subplots(total_rows, cols, figsize=(18, 5 * total_rows))
    
    # Flatten axes array for easy indexing, handling single year/row cases
    if total_rows * cols == 1:
        axes = np.array([axes])
    elif total_rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()


    print(f"--- Analysis of Extreme Price Hours (P1 and P99) from {years[0]} to {years[-1]} ---")
    
    # --- 3. Iterate, Calculate Percentiles, and Plot ---
    for i, year in enumerate(years):
        df_year = df[df['Year'] == year].copy()
        
        if df_year.empty:
            print(f"Warning: No data found for the year {year}")
            continue

        # Calculate the 1st and 99th percentiles for the year
        # <<< CHANGE 1: Updated quantiles to 0.01 (P1) and 0.99 (P99) >>>
        p1 = df_year['Price'].quantile(0.01)
        p99 = df_year['Price'].quantile(0.99)
        
        # Identify extreme hours (troughs and peaks)
        # <<< CHANGE 2: Updated labels to P99 Peak and P1 Trough >>>
        df_year['Is_Extreme'] = np.where(df_year['Price'] >= p99, 'P99 Peak', 
                                         np.where(df_year['Price'] <= p1, 'P1 Trough', 'Normal'))
        
        df_extreme = df_year[df_year['Is_Extreme'] != 'Normal'].sort_values(by='Price', ascending=False)
        
        # Store the extreme hours data (Raw Data Output)
        output_data = df_extreme[['Price', 'Year', 'Month', 'Day', 'Hour', 'Is_Extreme']].reset_index(drop=True)
        all_extreme_hours.append(output_data)
        
        # --- Plotting Setup for the Year ---
        
        hour_ax_index = i * rows_per_year
        month_ax_index = i * rows_per_year + 1

        # Adjust index based on column wrap
        ax1_idx = (hour_ax_index // cols) * cols + (hour_ax_index % cols)
        ax2_idx = (month_ax_index // cols) * cols + (month_ax_index % cols)

        # Use the correct axes based on the total row/column structure
        ax1 = axes[ax1_idx]
        ax2 = axes[ax2_idx]

        # Define peak/trough data for plotting
        df_peak = df_extreme[df_extreme['Is_Extreme'] == 'P99 Peak']
        df_trough = df_extreme[df_extreme['Is_Extreme'] == 'P1 Trough']
        
        # ------------------- PLOT 1: Hour vs Price -------------------
        ax1.scatter(df_year['Hour'], df_year['Price'], 
                    label='All Hours', alpha=0.15, color='gray', s=5)
        
        # <<< CHANGE 3: Updated label text to 99th Percentile Peak >>>
        ax1.scatter(df_peak['Hour'], df_peak['Price'], 
                    label=f'99th Percentile Peak (Price > {p99:.2f})', 
                    color='red', marker='^', s=50, zorder=5)
                    
        # <<< CHANGE 3: Updated label text to 1st Percentile Trough >>>
        ax1.scatter(df_trough['Hour'], df_trough['Price'], 
                    label=f'1st Percentile Trough (Price < {p1:.2f})', 
                    color='darkgreen', marker='v', s=50, zorder=5)

        ax1.set_title(f'Price Extremes by Hour - {year}', fontsize=14)
        ax1.set_xlabel('Hour (0-23)', fontsize=12)
        ax1.set_ylabel('Price (€/MWh)', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend(loc='upper left')
        ax1.set_xlim(-0.5, 23.5)
        
        # ------------------- PLOT 2: Month of Year vs Price -------------------
        ax2.scatter(df_year['Month'], df_year['Price'], 
                    label='All Hours', alpha=0.15, color='gray', s=5)
        
        # <<< CHANGE 3: Updated label text to 99th Percentile Peak >>>
        ax2.scatter(df_peak['Month'], df_peak['Price'], 
                    label=f'99th Percentile Peak (Price > {p99:.2f})', 
                    color='red', marker='^', s=50, zorder=5)
                    
        # <<< CHANGE 3: Updated label text to 1st Percentile Trough >>>
        ax2.scatter(df_trough['Month'], df_trough['Price'], 
                    label=f'1st Percentile Trough (Price < {p1:.2f})', 
                    color='darkgreen', marker='v', s=50, zorder=5)

        ax2.set_title(f'Price Extremes by Month of Year - {year}', fontsize=14)
        ax2.set_xlabel('Month of Year (1=Jan, 12=Dec)', fontsize=12)
        ax2.set_ylabel('Price (€/MWh)', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.set_xticks(range(1, 13))
        ax2.legend(loc='upper left')
        ax2.set_xlim(0.5, 12.5) # Set range from 1 to 12

        # Set consistent Y-axis limits across the year's plots for comparison
        price_min = df_year['Price'].min() - 5
        price_max = df_year['Price'].max() + 5
        ax1.set_ylim(price_min, price_max)
        ax2.set_ylim(price_min, price_max)
        
        
    # Hide any unused subplots
    num_plots_needed = len(years) * rows_per_year
    for j in range(num_plots_needed, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    
    # --- Save the plot to the specified output folder ---
    # <<< CHANGE 4: Updated output filenames for clarity >>>
    plot_save_path = os.path.join(output_folder, 'Price_Extremes_Hour_and_Month_P1_P99.png')
    plt.savefig(plot_save_path)
    print(f"\n--- Plot saved to: {plot_save_path} ---")
    
    # --- Save combined output to Excel in the specified output folder ---
    if all_extreme_hours:
        combined_output_df = pd.concat(all_extreme_hours)
        excel_save_path = os.path.join(output_folder, 'Extreme_Price_Hours_Output_P1_P99.xlsx')
        combined_output_df.to_excel(excel_save_path, index=False)
        print(f"--- Combined extreme hours saved to: {excel_save_path} ---")
    else:
        print("Warning: No extreme hours data was collected to save.")


# --- EXECUTION ---
file_to_analyze = r"C:\Users\harsh\OneDrive\Documents\Master thesis\Raw data\Finalized 2024 data\Dataset_germany_fossil_fuels_with_2024.csv"

analyze_and_plot_extreme_prices(file_to_analyze)