import pandas as pd
import os
import glob
# Removed: from tqdm import tqdm # Library for progress bar

# --- Configuration ---

# Input Folder: Contains the daily CSV files (auction_aggregated_curves_...)
INPUT_DIR = r"C:\Users\harsh\OneDrive\Documents\Master thesis\Raw data\Finalized 2024 data\germany_luxembourg_2024"

# Output Folder: Where the final Excel summary will be saved
OUTPUT_DIR = r"C:\Users\harsh\OneDrive\Documents\Master thesis\Raw data\Finalized 2024 data\Results\Part 2\Patterns 2024"
OUTPUT_FILENAME = "average_bid_patterns_2024.xlsx"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

# Price Ranges to Analyze (Lower Bound, Upper Bound, Display Name)
PRICE_RANGES = [
    (30, 100, "30-100"),
    (100, 200, "100-200"),
    (200, 400, "200-400"),
    (400, 600, "400-600"),
    (600, 800, "600-800")
]

# Hours to Analyze (Column D in CSV)
SELECTED_HOURS = [8, 11, 14, 17, 20]

# --- Main Logic ---

def get_volume_at_price(df_hourly, price_target):
    """
    Finds the cumulative volume corresponding to the first bid price 
    that is greater than or equal to the price_target.
    
    Args:
        df_hourly (pd.DataFrame): DataFrame filtered for a single hour/day.
        price_target (float): The target price boundary (€/MWh).
        
    Returns:
        float: The cumulative volume at or above the target price, or 0.0 if not found.
    """
    # 1. Filter for bids >= the target price
    df_filtered = df_hourly[df_hourly['Price [EUR/MWh]'] >= price_target]
    
    if df_filtered.empty:
        # If no bids are above the target, the volume is 0
        return 0.0
    
    # 2. The cumulative volume column is expected to be sorted ascending by price.
    #    We take the volume associated with the lowest price that meets the criterion.
    #    Since the Volume is cumulative, this is the volume *at that price point*.
    return df_filtered.iloc[0]['Cumulative Volume [MWh]']


def process_daily_file(file_path):
    """
    Loads, cleans, filters, and analyzes one daily CSV file to extract volumes 
    for the specified price ranges and hours.
    
    Returns:
        pd.DataFrame or None: A DataFrame containing the calculated volumes for the day.
    """
    try:
        # 1. Load the CSV, skipping the first two rows (header=2)
        # We explicitly set the column names based on the instruction (D, E, F, G)
        # assuming the standard EPEX file structure where these columns are D, E, F, G
        df = pd.read_csv(
            file_path, 
            header=2, 
            encoding='utf-8' # Use standard encoding
        )
        
        # Rename columns for clarity based on user description (D, E, F, G)
        # Assuming 0-indexed columns for:
        # 3: AuctionHour (D)
        # 4: Price [EUR/MWh] (E)
        # 5: Cumulative Volume [MWh] (F)
        # 6: Bid / Offer Type (G)
        df.rename(columns={
            df.columns[3]: 'AuctionHour',
            df.columns[4]: 'Price [EUR/MWh]',
            df.columns[5]: 'Cumulative Volume [MWh]',
            df.columns[6]: 'Bid / Offer Type'
        }, inplace=True)
        
        # 2. Filter for only "Sell" type bids (Column G)
        df_sell = df[df['Bid / Offer Type'] == 'Sell'].copy()
        
        if df_sell.empty:
            return None

        # Ensure numeric types
        df_sell['Price [EUR/MWh]'] = pd.to_numeric(df_sell['Price [EUR/MWh]'], errors='coerce')
        df_sell['Cumulative Volume [MWh]'] = pd.to_numeric(df_sell['Cumulative Volume [MWh]'], errors='coerce')
        
        # Drop rows where price or volume failed to convert
        df_sell.dropna(subset=['Price [EUR/MWh]', 'Cumulative Volume [MWh]'], inplace=True)

        # 3. Analyze data for each selected hour
        hourly_results = []
        
        for hour in SELECTED_HOURS:
            df_hourly = df_sell[df_sell['AuctionHour'] == hour].sort_values(
                by='Price [EUR/MWh]', ascending=True
            ).reset_index(drop=True)
            
            if df_hourly.empty:
                continue

            for lower_bound, upper_bound, range_name in PRICE_RANGES:
                # Find cumulative volume at the upper and lower bound
                vol_at_upper = get_volume_at_price(df_hourly, upper_bound)
                vol_at_lower = get_volume_at_price(df_hourly, lower_bound)
                
                # Calculate the volume *in* the range by subtraction
                volume_in_range = vol_at_upper - vol_at_lower

                hourly_results.append({
                    'Date': os.path.basename(file_path).split('.')[0].split('_')[-1], # Extract YYYYMMDD
                    'Hour': hour,
                    'Bid Price range': range_name,
                    'Volume Offered [MW]': max(0, volume_in_range) # Ensure non-negative volume
                })
        
        return pd.DataFrame(hourly_results)
        
    except Exception as e:
        print(f"Error processing file {os.path.basename(file_path)}: {e}")
        return None


def run_analysis():
    """Main function to orchestrate the processing of all files and saving the result."""
    
    print(f"Starting analysis of bids in: {INPUT_DIR}")
    
    # 1. Get all CSV files in the input directory
    all_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    
    if not all_files:
        print(f"Error: No CSV files found in {INPUT_DIR}. Please check the path.")
        return

    # 2. Process all files and concatenate results
    all_results = []
    total_files = len(all_files)
    
    # Use simple file counter instead of tqdm
    for i, file_path in enumerate(all_files):
        print(f"Processing daily bid curves: {i + 1}/{total_files} ({os.path.basename(file_path)})")
        daily_df = process_daily_file(file_path)
        if daily_df is not None:
            all_results.append(daily_df)

    if not all_results:
        print("Analysis complete, but no valid data was processed. Check file format/content.")
        return
        
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # 3. Calculate the required averages
    print("\nCalculating average volumes offered...")
    
    # Group by Hour and Price Range, then calculate the mean volume
    avg_df = combined_df.groupby(['Bid Price range', 'Hour'])['Volume Offered [MW]'].mean().reset_index()
    
    # Pivot the table to get the desired Excel format (Hours as columns)
    pivot_table = avg_df.pivot(
        index='Bid Price range', 
        columns='Hour', 
        values='Volume Offered [MW]'
    ).reset_index()
    
    # Rename columns to match the user's requested display format
    pivot_table.columns.name = None # Remove the 'Hour' name from the axis
    pivot_table.rename(columns={col: f'Hour {col}' for col in SELECTED_HOURS}, inplace=True)
    
    # 4. Final formatting and saving to Excel
    
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Sort the index/ranges to appear in the correct order (30-100, 100-200, etc.)
    range_order = [r[2] for r in PRICE_RANGES]
    pivot_table['Bid Price range'] = pd.Categorical(pivot_table['Bid Price range'], categories=range_order, ordered=True)
    pivot_table.sort_values('Bid Price range', inplace=True)
    
    pivot_table.to_excel(
        OUTPUT_FILE, 
        index=False, 
        sheet_name='Average Bids'
    )

    print("\n--- ANALYSIS COMPLETE ---")
    print(f"Average bid volumes successfully saved to:")
    print(f"{OUTPUT_FILE}")


# --- Execution ---
if __name__ == "__main__":
    run_analysis()