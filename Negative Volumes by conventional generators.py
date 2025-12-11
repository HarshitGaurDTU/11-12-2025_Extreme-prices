import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# --- 1. Define File Paths ---
input_file_path = r'C:\Users\harsh\OneDrive\Documents\Master thesis\Raw data\Finalized 2024 data\Non Standarized_combined.xlsx'
output_dir = r'C:\Users\harsh\OneDrive\Documents\Master thesis\Raw data\Finalized 2024 data\Results\Part 2\Model'
output_filename = 'net_negative_bid_volume_hourly_comparison_final.xlsx' # Analysis output file
output_file_path = os.path.join(output_dir, output_filename)
plot_dir = os.path.join(output_dir, 'Plots') # Directory for saving plots

# Ensure output directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)
print("--- Starting Hourly Comparison Analysis and Plotting ---")

## --- 2. Load Data ---
try:
    df = pd.read_excel(input_file_path)
except FileNotFoundError:
    print(f"Error: Input file not found at {input_file_path}")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred while reading the Excel file: {e}")
    sys.exit(1)

# List of columns to check for in the DataFrame
required_cols = ['Year', 'Month', 'Day', 'Hour', 'Price_Range', 'Aggregated Volume', 'Solar', 'Total Wind', 'Extreme hour', 'Price']
if not all(col in df.columns for col in required_cols):
    missing_cols = [col for col in required_cols if col not in df.columns]
    print(f"Error: DataFrame is missing one or more required columns: {missing_cols}")
    sys.exit(1)

if df['Year'].dtype not in ['int64', 'int32']:
    df['Year'] = df['Year'].astype(int)


# --- Calculation Functions ---

def calculate_net_negative_bid_volume(df_year):
    """Calculates Net Negative Bid Volume and adds the Price_Sign using the actual Price column."""

    negative_price_ranges = ['-501 to -150', '-150 to -20', '-20 to 0']
    negative_bids_mask = df_year['Price_Range'].isin(negative_price_ranges)
    df_negative_bids = df_year[negative_bids_mask].copy()

    negative_volume_sum = df_negative_bids.groupby(['Year', 'Month', 'Day', 'Hour'])['Aggregated Volume'].sum().reset_index()
    negative_volume_sum = negative_volume_sum.rename(columns={'Aggregated Volume': 'Negative_Bid_Volume'})

    df_hourly_info = df_year[['Year', 'Month', 'Day', 'Hour', 'Solar', 'Total Wind', 'Extreme hour', 'Price']].drop_duplicates()
    
    df_combined = pd.merge(negative_volume_sum, df_hourly_info, 
                             on=['Year', 'Month', 'Day', 'Hour'], 
                             how='right')
    
    df_combined['Negative_Bid_Volume'] = df_combined['Negative_Bid_Volume'].fillna(0)
    df_combined['Solar'] = df_combined['Solar'].fillna(0)
    df_combined['Total Wind'] = df_combined['Total Wind'].fillna(0)

    df_combined['Net_Negative_Bid_Volume'] = (
        df_combined['Negative_Bid_Volume'] - 
        df_combined['Solar'] - 
        df_combined['Total Wind']
    )

    # Use actual Price to determine Price_Sign: Price > 0 (1), Price <= 0 (-1)
    df_combined['Price_Sign'] = df_combined['Price'].apply(lambda x: 1 if x > 0 else -1)
    
    return df_combined


def calculate_hourly_averages(df_results, condition_name, extreme_hour_val, price_sign_val=None):
    """Calculates the 24 hourly averages for a specific condition."""
    df_filtered = df_results[df_results['Extreme hour'] == extreme_hour_val].copy()
    
    if price_sign_val is not None:
        df_filtered = df_filtered[df_filtered['Price_Sign'] == price_sign_val]
        
    hourly_avg = df_filtered.groupby('Hour')['Net_Negative_Bid_Volume'].mean().reset_index()
    hourly_avg.rename(columns={'Net_Negative_Bid_Volume': 'Average_Volume'}, inplace=True)
    
    return hourly_avg


# --- 3. Main Logic for Iterating and Collecting Data ---
analysis_years = range(2019, 2025) 
all_comparison_dataframes = [] # For Excel output
non_extreme_hourly_averages = [] # For Plot 2 (2019-2024 trend)

print("\n--- Processing Data by Year (2019-2024) ---")
for year in analysis_years:
    df_year = df[df['Year'] == year].copy()
    
    if df_year.empty:
        print(f"Skipping Year {year}: No data found.")
        continue
    
    df_results = calculate_net_negative_bid_volume(df_year)

    # Calculate 24 hourly averages for the three conditions
    avg_non_extreme = calculate_hourly_averages(df_results, 'Non-Extreme', 0)
    avg_extreme_pos = calculate_hourly_averages(df_results, 'Extreme positive', 1, price_sign_val=1)
    avg_extreme_neg = calculate_hourly_averages(df_results, 'Extreme negative', 1, price_sign_val=-1)

    # Prepare for Excel Output
    hours_template = pd.DataFrame({'Hour': range(1, 25)})

    non_extreme_data = hours_template.merge(avg_non_extreme, on='Hour', how='left').set_index('Hour').rename(columns={'Average_Volume': 'Non-Extreme'})
    extreme_pos_data = hours_template.merge(avg_extreme_pos, on='Hour', how='left').set_index('Hour').rename(columns={'Average_Volume': 'Extreme positive'})
    extreme_neg_data = hours_template.merge(avg_extreme_neg, on='Hour', how='left').set_index('Hour').rename(columns={'Average_Volume': 'Extreme negative'})
    
    comparison_df_year = pd.concat([non_extreme_data, extreme_pos_data, extreme_neg_data], axis=1).T
    comparison_df_year.columns.name = 'Hour'
    comparison_df_year.index.name = 'Extreme Type'
    
    all_comparison_dataframes.append({
        'year': year,
        'df': comparison_df_year
    })

    # Prepare data for Plot 2 (Non-Extreme Trend)
    avg_non_extreme.rename(columns={'Average_Volume': year}, inplace=True)
    non_extreme_hourly_averages.append(avg_non_extreme.set_index('Hour'))

    # Store 2024 data specifically for Plot 1
    if year == 2024:
        df_2024_plot = comparison_df_year.copy()


# --- 4. Save Results to Excel ---
if not all_comparison_dataframes:
    print("\nError: No data was processed for any year. Cannot save results.")
    sys.exit(1)

try:
    with pd.ExcelWriter(output_file_path, engine='xlsxwriter') as writer:
        for item in all_comparison_dataframes:
            sheet_name = f'Year_{item["year"]}'
            item['df'].to_excel(writer, sheet_name=sheet_name, index=True)
            
    print(f"\n\n✅ Hourly comparison results saved to: {output_file_path}")
except Exception as e:
    print(f"\nError saving file: {e}")


## --- 5. Plotting ---

# 5a. Plot 1: 2024 Hourly Comparison (3 Cases)
if 'df_2024_plot' in locals():
    plt.figure(figsize=(12, 6))
    
    # Plot each of the three conditions for 2024
    for condition in df_2024_plot.index:
        plt.plot(df_2024_plot.columns, df_2024_plot.loc[condition], label=condition, marker='.')

    plt.title('2024 Hourly Average Net Negative Bid Volume: Non-Extreme vs. Extreme Hours', fontsize=14)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Average Net Negative Bid Volume (MWh)', fontsize=12)
    plt.xticks(range(1, 25)) 
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Condition')
    plt.tight_layout()

    plot1_path = os.path.join(plot_dir, '2024_hourly_comparison_plot.png')
    plt.savefig(plot1_path)
    print(f"✅ Plot 1 (2024 Comparison) saved to: {plot1_path}")
    plt.show()
else:
    print("Warning: Data for year 2024 was not found. Skipping Plot 1.")


# 5b. Plot 2: Non-Extreme Hourly Trend (2019-2024)
if non_extreme_hourly_averages:
    combined_non_extreme = pd.concat(non_extreme_hourly_averages, axis=1)

    plt.figure(figsize=(12, 6))

    # Plot each year's non-extreme average Net Negative Bid Volume
    for year in combined_non_extreme.columns:
        plt.plot(combined_non_extreme.index, combined_non_extreme[year], label=f'Year {year}')

    plt.title('Non-Extreme Hourly Average Net Negative Bid Volume Trend (2019-2024)', fontsize=14)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Average Net Negative Bid Volume (MWh)', fontsize=12)
    plt.xticks(range(1, 25))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Year')
    plt.tight_layout()

    plot2_path = os.path.join(plot_dir, 'non_extreme_hourly_trend_plot.png')
    plt.savefig(plot2_path)
    print(f"✅ Plot 2 (Non-Extreme Trend) saved to: {plot2_path}")
    plt.show()
else:
    print("Warning: No non-extreme data found for plotting. Skipping Plot 2.")

print("\n--- Script Execution Complete ---")