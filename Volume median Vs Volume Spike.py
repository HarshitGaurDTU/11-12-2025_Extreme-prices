import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# --- Configuration ---

# NOTE: The script is designed to run in a controlled environment.
# Please ensure the input file path is accessible in your local environment.
INPUT_FILEPATH = r'C:\Users\harsh\OneDrive\Documents\Master thesis\Raw data\Finalized 2024 data\Non Standarized_combined.xlsx'

# New configuration for plot output directory and filename
OUTPUT_PLOT_DIR = r'C:\Users\harsh\OneDrive\Documents\Master thesis\Raw data\Finalized 2024 data\Results\Part 2'
OUTPUT_PLOT_FILENAME = 'hourly_volume_analysis.png'


# Define the groups of Price_Range columns for volume summation
# Group 1: Volumes offered at Negative Prices (Low/High)
PRICE_GROUP_NEGATIVE = [
    '-501 to -150', '-150 to -20', '-20 to 0'
]
# Group 2: Volumes offered up to Low Positive Prices (covers all negative plus low positive)
PRICE_GROUP_POSITIVE = PRICE_GROUP_NEGATIVE + [
    '0 to 30', '30 to 80', '80 to 150'
]

# Required columns from the input file
REQUIRED_COLS = [
    'Year', 'Month', 'Day', 'Hour', 'Price_Range', 
    'Aggregated Volume', 'Price', 'Extreme hour'
]

def load_data(filepath):
    """Loads the data and ensures required columns are present."""
    if not os.path.exists(filepath):
        print(f"Error: File not found at '{filepath}'.")
        # Returning an empty DataFrame to allow script to exit gracefully
        return pd.DataFrame() 

    print(f"Loading data from: {filepath}")
    try:
        df = pd.read_excel(filepath)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return pd.DataFrame()
    
    # Check for required columns
    missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns in the dataset: {missing_cols}")
        return pd.DataFrame()

    df = df[REQUIRED_COLS].copy()
    print(f"Data loaded successfully. Shape: {df.shape}")
    
    # Ensure numerical columns are correctly typed
    for col in ['Year', 'Hour', 'Aggregated Volume', 'Price', 'Extreme hour']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df


def calculate_volumes(df):
    """
    Calculates the summed volume for the Negative and Positive Price Groups 
    for every unique (Year, Month, Day, Hour) timestamp.
    """
    print("Calculating daily/hourly summed volumes...")
    
    # 1. Initialize volume columns
    df['Negative_Price_Group_Volume'] = 0.0
    df['Positive_Price_Group_Volume'] = 0.0

    # 2. Filter data for the two groups
    # Negative Group volume is only counted if Price_Range is in the negative group
    neg_mask = df['Price_Range'].isin(PRICE_GROUP_NEGATIVE)
    df.loc[neg_mask, 'Negative_Price_Group_Volume'] = df.loc[neg_mask, 'Aggregated Volume']

    # Positive Group volume is only counted if Price_Range is in the positive group
    pos_mask = df['Price_Range'].isin(PRICE_GROUP_POSITIVE)
    df.loc[pos_mask, 'Positive_Price_Group_Volume'] = df.loc[pos_mask, 'Aggregated Volume']

    # 3. Sum volumes for the two groups by unique time point (Year, Month, Day, Hour)
    
    # Aggregate to get the total volume contributed by the respective price ranges 
    # for each unique hourly observation (Day of the year)
    volume_sums = df.groupby(['Year', 'Month', 'Day', 'Hour']).agg(
        Total_Neg_Vol=('Negative_Price_Group_Volume', 'sum'),
        Total_Pos_Vol=('Positive_Price_Group_Volume', 'sum'),
        Extreme_hour=('Extreme hour', 'max'),
        Price=('Price', 'mean'), # Take the mean price for spike identification later
    ).reset_index()
    
    return volume_sums


def calculate_median_and_spike_data(df_sums):
    """
    Calculates the median volumes (non-extreme) and extracts spike volumes (extreme).
    """
    print("Calculating median and spike volumes...")
    
    # --- 1. Calculate Median Volumes (Extreme hour == 0) ---
    df_median = df_sums[df_sums['Extreme_hour'] == 0]
    
    # Calculate median of the summed volumes per (Year, Hour)
    df_median_hourly = df_median.groupby(['Year', 'Hour']).agg(
        Median_Neg_Vol=('Total_Neg_Vol', 'median'),
        Median_Pos_Vol=('Total_Pos_Vol', 'median')
    ).reset_index()

    # --- 2. Extract Spike Volumes (Extreme hour == 1) ---
    df_spike = df_sums[df_sums['Extreme_hour'] == 1].copy()
    
    # Identify negative and positive spike events based on the Price at that hour
    # Note: We use the mean price calculated in the volume_sums step
    df_spike['Spike_Type'] = np.where(df_spike['Price'] < 0, 
                                     'Negative_Spike', 
                                     'Positive_Spike')
    
    # Select the relevant volume for each spike type
    # Negative spike volume uses Total_Neg_Vol, Positive spike uses Total_Pos_Vol
    df_spike['Spike_Volume'] = np.where(df_spike['Spike_Type'] == 'Negative_Spike', 
                                       df_spike['Total_Neg_Vol'], 
                                       df_spike['Total_Pos_Vol'])
    
    return df_median_hourly, df_spike

def plot_analysis(df_median, df_spike):
    """
    Creates a multi-panel plot, one for each year, comparing median and spike volumes.
    Also saves the final plot to the configured output path.
    """
    years = sorted(df_median['Year'].unique())
    num_years = len(years)

    if num_years == 0:
        print("No data available for plotting.")
        return

    # Determine subplot layout: Aim for a square or slightly vertical grid
    rows = int(np.ceil(np.sqrt(num_years)))
    cols = int(np.ceil(num_years / rows))
    
    # Set up the figure and axes with smaller figsize (6x4 per subplot)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), 
                             sharex=True, sharey=False)
    
    # Flatten axes array for easy iteration if it's 2D
    if num_years > 1:
        axes = axes.flatten()
    elif num_years == 1:
        axes = [axes] # Ensure it's iterable

    print(f"Generating {num_years} plots in a {rows}x{cols} grid...")

    for i, year in enumerate(years):
        ax = axes[i]
        
        # 1. Filter data for the current year
        year_median = df_median[df_median['Year'] == year]
        year_spike = df_spike[df_spike['Year'] == year]
        
        # Split spikes by type
        neg_spikes = year_spike[year_spike['Spike_Type'] == 'Negative_Spike']
        pos_spikes = year_spike[year_spike['Spike_Type'] == 'Positive_Spike']
        
        # 2. Plot Median Lines
        # Negative Median Volume
        ax.plot(year_median['Hour'], year_median['Median_Neg_Vol'], 
                label='Negative Median Vol Offered', color='#e377c2', linewidth=2)
        
        # Positive Median Volume
        ax.plot(year_median['Hour'], year_median['Median_Pos_Vol'], 
                label='Positive Median Vol Offered', color='#1f77b4', linewidth=2)
        
        # 3. Plot Spike Events
        # Negative Spike Events (Total_Neg_Vol)
        ax.scatter(neg_spikes['Hour'], neg_spikes['Spike_Volume'], 
                   label='Negative Spike Event Vol', color='red', marker='X', s=50, zorder=5)
        
        # Positive Spike Events (Total_Pos_Vol)
        ax.scatter(pos_spikes['Hour'], pos_spikes['Spike_Volume'], 
                   label='Positive Spike Event Vol', color='green', marker='o', s=50, zorder=5)
        
        # 4. Customization
        ax.set_title(f'Year {year}: Hourly Median Volume vs. Spike Events', fontsize=12) # Reduced font size for less clutter
        ax.set_xlabel('Hour of Day', fontsize=10)
        ax.set_ylabel('Aggregated Volume (MW/MWh)', fontsize=10)
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Dynamic legend: only show main legend on the first subplot
        if i == 0:
             ax.legend(loc='upper left', fontsize=8)

        # Optional: Hypothesis text removed to save space, but kept mean calculation
        mean_neg_spike = neg_spikes['Spike_Volume'].mean()
        mean_pos_spike = pos_spikes['Spike_Volume'].mean()
        

    # Hide any unused subplots
    for j in range(num_years, rows * cols):
        fig.delaxes(axes[j])
        
    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to make space for suptitle
    fig.suptitle('Annual Hourly Volume Analysis: Median vs. Extreme Spikes', fontsize=14, fontweight='bold') # Reduced font size for better fit
    
    # --- Saving Plot Logic ---
    full_plot_path = os.path.join(OUTPUT_PLOT_DIR, OUTPUT_PLOT_FILENAME)
    print(f"\nSaving plot to: {full_plot_path}")
    
    # Ensure directory exists
    os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
    
    try:
        # Save the figure with high resolution
        plt.savefig(full_plot_path, dpi=300)
        print("Plot saved successfully.")
    except Exception as e:
        print(f"ERROR: Failed to save plot. Reason: {e}")
        
    plt.show()


# --- Main Execution ---
if __name__ == '__main__':
    # 1. Load Data
    df = load_data(INPUT_FILEPATH)
    
    if df.empty:
        print("Script terminated due to error or no data loaded.")
    else:
        # 2. Calculate Summed Volumes per unique hour (Day-level)
        df_sums = calculate_volumes(df)
        
        # 3. Calculate Hourly Median and Extract Spike Events
        df_median_hourly, df_spike_events = calculate_median_and_spike_data(df_sums)
        
        # 4. Plot Results
        plot_analysis(df_median_hourly, df_spike_events)
        print("\nAnalysis and plotting complete.")