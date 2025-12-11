import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# --- 1. Define Paths and Parameters ---
base_path = r'C:\Users\harsh\OneDrive\Documents\Master thesis\Raw data\Finalized 2024 data'
input_folder = os.path.join(base_path, 'germany_luxembourg_2024')
output_folder = os.path.join(base_path, r'Results\Part 2\Patterns 2024\Dec 2024')

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Files and Day of Interest
target_file_dates = ['20241211', '20241212', '20241213', '20240825'] # ADDED 20240825
SINGLE_DAY_OF_INTEREST = '20241212'
NEW_DAY_OF_INTEREST = '20240825' # New date for Plot 6
file_pattern = 'auction_aggregated_curves_germany_luxembourg_{}.csv'
PRICE_CAP = 1000.0

# Columns for data processing
COLUMN_MAP = {
    'Date': 'Date',         
    'Hour': 'Hour',         
    'Price': 'Price',       
    'Volume': 'Volume',     
    'Type_Filter': 'Sale/Purchase' 
}
ALL_COLUMN_NAMES = ['Date', 'Week', 'Weekday', 'Hour', 'Price', 'Volume', 'Sale/Purchase']

# Define the hours needed for all 6 plots
hours_needed = set()
hours_needed.update([17, 8]) 
hours_needed.update([15, 16, 17, 18, 19]) 
hours_needed.update([6, 7, 8, 9]) 
hours_needed.update([3, 4, 5]) # ADDED HOURS for Plot 6
all_required_hours = list(hours_needed)


# --- 2. Data Loading and Preprocessing ---
all_data = []

print("Starting data loading and filtering...")
for date_str in target_file_dates:
    file_path = os.path.join(input_folder, file_pattern.format(date_str))
        
    try:
        df = pd.read_csv(
            file_path, 
            header=None, 
            skiprows=2, 
            names=ALL_COLUMN_NAMES
        )

        for col in [COLUMN_MAP['Price'], COLUMN_MAP['Volume']]:
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df[COLUMN_MAP['Hour']] = pd.to_numeric(df[COLUMN_MAP['Hour']], errors='coerce').astype('Int64')

        df.dropna(subset=[COLUMN_MAP['Hour'], COLUMN_MAP['Price'], COLUMN_MAP['Volume']], inplace=True)

        df_filtered = df[
            (df[COLUMN_MAP['Hour']].isin(all_required_hours)) & 
            (df[COLUMN_MAP['Type_Filter']].astype(str).str.contains('Sell', na=False))
        ].copy()
        
        df_filtered['Day'] = date_str
        all_data.append(df_filtered[['Day', COLUMN_MAP['Hour'], COLUMN_MAP['Price'], COLUMN_MAP['Volume']]])

    except Exception as e:
        print(f"An unexpected error occurred while processing {date_str}. Error: {e}")
        continue

if not all_data:
    print("No data was successfully loaded or filtered. Exiting.")
    exit() 

combined_df_full = pd.concat(all_data, ignore_index=True)

PRICE_COL = COLUMN_MAP['Price']
VOLUME_COL = COLUMN_MAP['Volume']
HOUR_COL = COLUMN_MAP['Hour']

# Apply the global price cap filter
combined_df_capped = combined_df_full[combined_df_full[PRICE_COL] <= PRICE_CAP].copy()
print("Data loading and filtering complete (Price capped at €1000/MWh).")


# --- 3. Single-Panel Plotting Function ---

def create_single_panel_plot(data_frame, plot_title, filename_suffix, plot_mode, target_hour=None, target_hours_list=None, target_days_list=None,
price_min=None, price_max=PRICE_CAP):
    """
    Generates a single-panel plot comparing either:
    - 'cross_day': A single hour across multiple days (target_hour, target_days_list)
    - 'single_day': Multiple hours on a single day (target_hours_list, target_days_list must be a single day)
    """
    plot_price_min = -20.0 if price_min is None else price_min
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # 1. Filter data based on mode
    if plot_mode == 'cross_day':
        plot_df = data_frame[
            (data_frame[HOUR_COL] == target_hour) & 
            (data_frame['Day'].isin(target_days_list))
        ]
        items_to_plot = target_days_list
        label_key = 'Day'
        label_prefix = 'Day '
    
    elif plot_mode == 'single_day':
        single_day = target_days_list[0]
        plot_df = data_frame[
            (data_frame[HOUR_COL].isin(target_hours_list)) & 
            (data_frame['Day'] == single_day)
        ]
        items_to_plot = target_hours_list
        label_key = HOUR_COL
        label_prefix = 'Hour '
        
    else:
        print(f"Error: Invalid plot_mode '{plot_mode}'")
        plt.close(fig)
        return

    # 2. Apply requested price filters
    if price_min is not None:
        plot_df = plot_df[plot_df[PRICE_COL] >= plot_price_min]
    if price_max is not None:
        plot_df = plot_df[plot_df[PRICE_COL] <= price_max]

    if plot_df.empty:
        ax.text(0.5, 0.5, 'No Sell Data Found', transform=ax.transAxes, ha='center', va='center', fontsize=16, color='red')
        ax.set_title(plot_title)
        print(f"Warning: No data found for plot {filename_suffix}. Skipping.")
        plt.close(fig)
        return

    # 3. Determine overall volume range and plot
    min_volume = plot_df[VOLUME_COL].min() * 0.95 
    max_volume = plot_df[VOLUME_COL].max() * 1.05
    
    for item in items_to_plot:
        # Filter for the specific item (Day or Hour)
        item_df = plot_df[plot_df[label_key] == item]

        item_df = item_df.sort_values(by=PRICE_COL)
        
        ax.plot(item_df[VOLUME_COL], item_df[PRICE_COL], 
                marker='.', linestyle='-', label=f'{label_prefix}{item}', linewidth=1.5, markersize=3)
            
    # 4. Formatting
    ax.set_xlim(min_volume, max_volume)
    
    y_min_limit = 0 if price_min is None else price_min
    ax.set_ylim(plot_price_min, price_max)

    ax.set_title(plot_title)
    ax.set_ylabel("Price [€/MWh]")
    ax.set_xlabel(f"Aggregated Sell Volume [MWh] (Range {min_volume:.0f} - {max_volume:.0f})")
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend(loc='upper right', title=label_key if plot_mode == 'cross_day' else 'Hour')

    # 5. Save the plot
    plot_filename = os.path.join(output_folder, f'Supply_Stack_Comparison_{filename_suffix}.png')
    fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot '{plot_title}' saved successfully to: {plot_filename}")


# --- 4. Generate the 6 Requested Plots ---
print("\nGenerating the 6 requested single-panel plots...")

# 4a. Plot 1: Hour 17 across 3 days
create_single_panel_plot(
    combined_df_capped,
    "Plot 1: Cross-Day Supply Stack Comparison - Hour 17",
    "P1_H17_CrossDay",
    plot_mode='cross_day',
    target_hour=17,
    target_days_list=['20241211', '20241212', '20241213']
)

# 4b. Plot 2: Hours 15, 16, 17, 18, 19 on Day 20241212
create_single_panel_plot(
    combined_df_capped,
    f"Plot 2: Single-Day Supply Stack Comparison - Hours 15-19 on Day {SINGLE_DAY_OF_INTEREST}",
    "P2_H15to19_SingleDay",
    plot_mode='single_day',
    target_hours_list=[15, 16, 17, 18, 19],
    target_days_list=[SINGLE_DAY_OF_INTEREST]
)

# 4c. Plot 3: Hour 8 across 3 days
create_single_panel_plot(
    combined_df_capped,
    "Plot 3: Cross-Day Supply Stack Comparison - Hour 8",
    "P3_H08_CrossDay",
    plot_mode='cross_day',
    target_hour=8,
    target_days_list=['20241211', '20241212', '20241213']
)

# 4d. Plot 4: Hours 6, 7, 8, 9 on Day 20241212
create_single_panel_plot(
    combined_df_capped,
    f"Plot 4: Single-Day Supply Stack Comparison - Hours 6-9 on Day {SINGLE_DAY_OF_INTEREST}",
    "P4_H06to09_SingleDay",
    plot_mode='single_day',
    target_hours_list=[6, 7, 8, 9],
    target_days_list=[SINGLE_DAY_OF_INTEREST]
)

# 4e. Plot 5: Single-Day Comparison - Hours 8 and 17 (€30 to €1000)
create_single_panel_plot(
    combined_df_capped,
    f"Plot 5: Single-Day Comparison - Hours 8 & 17 on Day {SINGLE_DAY_OF_INTEREST} (€-20 to €1000)",
    "P5_H08_H17_Zoomed",
    plot_mode='single_day',
    target_hours_list=[8, 17],
    target_days_list=[SINGLE_DAY_OF_INTEREST],
    price_min=-20.0 
)

# 4f. Plot 6: NEW Single-Day Comparison - Hours 3, 4, 5 on Day 20240825
create_single_panel_plot(
    combined_df_capped,
    f"Plot 6: Single-Day Supply Stack Comparison - Hours 3-5 on Day {NEW_DAY_OF_INTEREST}",
    "P6_H02to05_SingleDay_NewDate",
    plot_mode='single_day',
    target_hours_list=[3, 4, 5],
    target_days_list=[NEW_DAY_OF_INTEREST]
)

print("\nAll 6 unique PNG plots have been successfully generated.")