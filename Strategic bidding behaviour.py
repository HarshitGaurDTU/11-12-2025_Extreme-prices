import pandas as pd
import numpy as np
import os
import warnings

def analyze_hourly_bid_volumes():
    """
    Analyzes hourly bid volumes based on custom price ranges and categorizes
    hours into 'Extreme positive', 'Extreme negative', and 'Non-Extreme' 
    using the existing 'Extreme hour' and 'Price' columns.

    The final output is an Excel file showing the average Aggregated Volume offered
    for each redefined bid range, split by hour and the three extreme cases, per year.
    """
    # Suppress warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # --- 1. CONFIGURATION ---
    INPUT_FILE_PATH = r'C:\Users\harsh\OneDrive\Documents\Master thesis\Raw data\Finalized 2024 data\Non Standarized_combined.xlsx'
    OUTPUT_DIR = r"C:\Users\harsh\OneDrive\Documents\Master thesis\Raw data\Finalized 2024 data\Results\Part 2\Model"
    OUTPUT_FILE_NAME = "Hourly_Average_Volume_by_Extreme_Case_Updated.xlsx"
    
    # Input columns
    PRICE_COL = 'Price'
    EXTREME_HOUR_COL = 'Extreme hour' # New required column
    YEAR_COL = 'Year'
    HOUR_COL = 'Hour'
    PRICE_RANGE_COL = 'Price_Range'
    VOLUME_COL = 'Aggregated Volume'
    
    # Target combined ranges for the final analysis output
    COMBINED_RANGES = {
        '"-500 to -20"': ['-501 to -150', '-150 to -20'],
        '"-20 to 30"': ['-20 to 0', '0 to 30'],
        '"30 to 80"': ['30 to 80'],
        '"80 to 150"': ['80 to 150'],
        '"150 to 250"': ['150 to 250'],
        '"250 to 500"': ['250 to 500'],
        '"500 to 1000"': ['500 to 1000'],
        '"1000 to 4001"': ['1000 to 4001'],
    }
    
    # Custom index structure for the final output (row headers)
    CASE_LABELS = ['Non-Extreme', 'Extreme positive', 'Extreme negative']
    
    # --- Check and Setup ---
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE_NAME)
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory: {e}")
        return

    print(f"Loading data from: {INPUT_FILE_PATH}...")
    
    try:
        df = pd.read_excel(INPUT_FILE_PATH)
    except FileNotFoundError:
        print(f"ERROR: Input file not found at {INPUT_FILE_PATH}")
        return
    except Exception as e:
        print(f"ERROR loading Excel file: {e}")
        return

    # Updated required columns list
    required_cols = [PRICE_COL, EXTREME_HOUR_COL, YEAR_COL, HOUR_COL, PRICE_RANGE_COL, VOLUME_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"ERROR: Missing required columns in the dataset: {missing_cols}")
        return

    # Ensure correct data types
    df[YEAR_COL] = df[YEAR_COL].astype(int)
    df[HOUR_COL] = df[HOUR_COL].astype(int)
    df[EXTREME_HOUR_COL] = df[EXTREME_HOUR_COL].astype(int) # Ensure Extreme hour is int
    df[VOLUME_COL] = pd.to_numeric(df[VOLUME_COL], errors='coerce')
    df[PRICE_COL] = pd.to_numeric(df[PRICE_COL], errors='coerce')
    df.dropna(subset=required_cols, inplace=True)
    
    if df.empty:
        print("ERROR: DataFrame is empty after dropping missing values.")
        return
        
    print(f"Data loaded successfully. Total rows: {len(df)}")

    # --- 2. IDENTIFY EXTREME CASES (UPDATED LOGIC) ---
    print("\nCategorizing hours based on 'Extreme hour' and 'Price' columns...")
    
    # Define conditions for categorization
    conditions = [
        # 1. Extreme positive: Extreme hour = 1 AND Price > 0
        (df[EXTREME_HOUR_COL] == 1) & (df[PRICE_COL] > 0),
        # 2. Extreme negative: Extreme hour = 1 AND Price <= 0
        (df[EXTREME_HOUR_COL] == 1) & (df[PRICE_COL] <= 0),
        # 3. Non-Extreme: Extreme hour = 0 (This covers everything else as the default)
        (df[EXTREME_HOUR_COL] == 0)
    ]
    
    # Define corresponding labels for the conditions
    choices = ['Extreme positive', 'Extreme negative', 'Non-Extreme']
    
    # Apply the categorization
    # Use np.select to efficiently apply the conditions in order
    df['Extreme_Case'] = np.select(conditions, choices, default='Non-Extreme')
    
    print(f"Distribution of Extreme Cases: \n{df['Extreme_Case'].value_counts()}")

    # --- 3. COMBINE PRICE RANGES ---
    print("\nMapping original price ranges to combined analysis ranges...")
    
    # Reverse mapping for efficient lookup: original_range -> combined_label
    reverse_map = {
        original_range: combined_label
        for combined_label, original_list in COMBINED_RANGES.items()
        for original_range in original_list
    }
    
    # Map the original Price_Range column to the new combined labels
    df['Combined_Range'] = df[PRICE_RANGE_COL].map(reverse_map)
    df.dropna(subset=['Combined_Range'], inplace=True)

    # --- 4. AGGREGATION AND PIVOTING ---
    print("Aggregating average volumes by Year, Hour, Extreme Case, and Combined Range...")
    
    # Group by the four analysis dimensions and calculate the MEAN of Aggregated Volume
    df_agg = df.groupby([YEAR_COL, HOUR_COL, 'Extreme_Case', 'Combined_Range'])[VOLUME_COL].mean().reset_index()
    
    # --- 5. RESTRUCTURING FOR FINAL EXCEL OUTPUT ---
    print("\nGenerating Excel output...")
    # Initialize Excel writer
    writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
    
    for year in sorted(df_agg[YEAR_COL].unique()):
        df_year_agg = df_agg[df_agg[YEAR_COL] == year].copy()
        
        # Create a multi-index for the final table rows (Combined_Range x Extreme_Case)
        midx = pd.MultiIndex.from_product([COMBINED_RANGES.keys(), CASE_LABELS], 
                                          names=['Average volumes', 'Extreme Type'])
        
        # Prepare the final structured DataFrame with 24 columns (for hours 0-23)
        df_final = pd.DataFrame(index=midx, columns=range(24))
        
        # Populate the final DataFrame
        for index, row in df_year_agg.iterrows():
            combined_range = row['Combined_Range']
            extreme_case = row['Extreme_Case']
            hour = row[HOUR_COL]
            average_volume = row[VOLUME_COL]
            
            # Populate the cell using the multi-index and hour column
            if hour in df_final.columns:
                 df_final.loc[(combined_range, extreme_case), hour] = average_volume

        # --- FINAL CLEANUP AND FORMATTING ---
        df_final = df_final.round(0)
        df_final.fillna('', inplace=True)
        
        # Write to Excel starting at row 2 (Excel Row 3)
        sheet_name = str(year)
        df_final.to_excel(writer, sheet_name=sheet_name, startrow=2)
        
        # Manual Excel formatting for headers
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]
        
        # Custom Formats
        header_format_year = workbook.add_format({'bold': True, 'bg_color': '#FFFF00', 'border': 1})
        header_format_merged = workbook.add_format({'align': 'center', 'bold': True})
        
        # A1: Year header (The yellow cell)
        worksheet.write('A1', sheet_name, header_format_year)
        
        # C2:Z2: Specific Hour >>>>> merged header (Excel Row 2, spanning columns above the hours 0-23)
        worksheet.merge_range('C2:Z2', 'Specific Hour >>>>>', header_format_merged)
        
        # Set column width for the index to show full range labels
        worksheet.set_column('A:B', 15)
        worksheet.set_column('C:Z', 10)
        
    # Save the Excel file
    writer.close()
    print(f"\n✅ Successfully generated and saved analysis to: {output_path}")

if __name__ == '__main__':
    analyze_hourly_bid_volumes()