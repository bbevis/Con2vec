import os
import pandas as pd
from datetime import datetime

# Function to log messages with timestamps
def log_message(message, level="INFO"):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] [{level}] {message}")

# Function to load a CSV file and handle errors
def load_csv(file_path):
    try:
        log_message(f"Loading file: {file_path}")
        return pd.read_csv(file_path)
    except Exception as e:
        log_message(f"Failed to load file: {file_path}. Error: {str(e)}", level="ERROR")
        return None

# Function to save a DataFrame to a CSV file
def save_csv(data, output_path):
    try:
        log_message(f"Saving merged data to: {output_path}")
        data.to_csv(output_path, index=False)
        log_message(f"Data successfully saved to: {output_path}")
    except Exception as e:
        log_message(f"Failed to save file: {output_path}. Error: {str(e)}", level="ERROR")

# Main function to perform the merging
def merge_files(base_file, text_file, vocal_file, output_file):
    log_message("Starting the merging process.")

    # Load files
    turn_ids = load_csv(base_file)
    text_agg = load_csv(text_file)
    vocal_agg = load_csv(vocal_file)

    # Check if all files loaded successfully
    if turn_ids is None or text_agg is None or vocal_agg is None:
        log_message("One or more files failed to load. Exiting process.", level="ERROR")
        return

    # Verify if the common key exists in all files
    common_key = "Pair_Speaker_turn"
    for file_name, df in zip(["TurnIDs.csv", "Text_agg.csv", "Vocal_agg.csv"], [turn_ids, text_agg, vocal_agg]):
        if common_key not in df.columns:
            log_message(f"Column '{common_key}' is missing in {file_name}. Exiting process.", level="ERROR")
            return

    # Merge the files
    try:
        text_columns_to_merge = [col for col in text_agg.columns if col != common_key]
        merged_data = pd.merge(turn_ids, text_agg, on=common_key, how="left", suffixes=('', '_text'))
        
        vvocal_columns_to_merge = [col for col in vocal_agg.columns if col != common_key]
        merged_data = pd.merge(merged_data, vocal_agg, on=common_key, how="left", suffixes=('', '_vocal'))
        
        log_message("Merging completed successfully.")
    except Exception as e:
        log_message(f"Error occurred during merging: {str(e)}", level="ERROR")
        return

    # Save the merged data
    save_csv(merged_data, output_file)
    log_message("Merging process completed.")

# Paths to the input and output files
base_file = os.path.join('Output', 'super_May22', 'TurnIDs.csv') # Base file
text_file = os.path.join('Output', 'super_May22', 'Text_agg.csv')  # Text aggregation file
vocal_file = os.path.join('Output', 'super_May22', 'Vocal_agg.csv')   # Vocal aggregation file
output_file = os.path.join('Output', 'super_May22', 'Merged_Data.csv') # Output file

# Execute the merging process
merge_files(base_file, text_file, vocal_file, output_file)
