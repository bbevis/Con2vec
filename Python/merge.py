import pandas as pd
from pathlib import Path
from datetime import datetime
from functools import reduce

# Configurable file paths
CONFIG = {
    "base": "my_output_chat_level.csv",
    "vocal": "Vocal_agg.csv",
    "visual": "Visual_agg.csv",
    "output": "multimodal_data.csv",
    "folder": Path("Output/super_May22")
}

file_paths = {key: CONFIG["folder"] / filename for key, filename in CONFIG.items() if key != "folder"}

# Logging function
def log_message(message, level="INFO"):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] [{level}] {message}")

# Load CSV function
def load_csv(file_path):
    try:
        log_message(f"Loading file: {file_path}")
        return pd.read_csv(file_path)
    except Exception as e:
        log_message(f"Failed to load file: {file_path}. Error: {e}", level="ERROR")
        return None

# Validate DataFrames
def validate_dataframe(df, file_name):
    if df is None or df.empty:
        log_message(f"Skipping {file_name}: File is empty or could not be loaded.", level="WARNING")
        return None
    if "Pair_Speaker_turn" not in df.columns:
        log_message(f"Skipping {file_name}: Missing 'Pair_Speaker_turn' column.", level="WARNING")
        return None
    return df

# Drop duplicate columns after merging
def remove_duplicate_columns(df):
    """ Remove duplicated columns created by merging (e.g., Turn Start_x, Turn Start_y). """
    for col in df.columns:
        if col.endswith('_x') or col.endswith('_y'):
            col_root = col[:-2]  # Remove suffix
            if col_root in df.columns:
                df.drop(columns=[col], inplace=True, errors='ignore')
            else:
                df.rename(columns={col: col_root}, inplace=True)
    return df

# Load and merge files
def merge_files():
    log_message("Starting multimodal merging process.")

    base_df = validate_dataframe(load_csv(file_paths["base"]), "base")
    if base_df is None:
        log_message("Base file is missing. Exiting.", level="ERROR")
        return

    dataframes = {key: validate_dataframe(load_csv(path), key) for key, path in file_paths.items() if key not in ["base", "output"]}
    dataframes = {k: v for k, v in dataframes.items() if v is not None}

    # Prevent duplicate columns before merging
    for key, df in dataframes.items():
        common_cols = set(base_df.columns) & set(df.columns) - {"Pair_Speaker_turn"}
        df.drop(columns=common_cols, errors='ignore', inplace=True)

    # Merge all DataFrames using reduce() and merge()
    if dataframes:
        merged_data = reduce(lambda left, right: pd.merge(left, right, on="Pair_Speaker_turn", how="left"), [base_df] + list(dataframes.values()))
    else:
        merged_data = base_df

    merged_data = remove_duplicate_columns(merged_data)  # Remove _x and _y suffixes

    merged_data.to_csv(file_paths["output"], index=False)
    log_message("Merging process completed.")

# Execute
merge_files()
