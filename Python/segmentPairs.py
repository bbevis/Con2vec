import pandas as pd
import numpy as np
import os
import logging

# Configure logging to output errors to a file and info to the console
logging.basicConfig(
    filename='stacking_errors.txt', 
    level=logging.ERROR, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_metadata(file_path: str) -> pd.DataFrame:
    """Load and filter metadata for audio files only."""
    try:
        metadata = pd.read_csv(file_path)
        return metadata[metadata['is_audio'] == 1]
    except Exception as e:
        logging.error("Error loading metadata: %s", e)
        raise

def stack_files_by_group(metadata: pd.DataFrame, base_directory: str, output_directory: str) -> pd.DataFrame:
    """
    Stacks files by SegmentID, adds metadata, and processes overlap and backchannel indicators.

    Args:
        metadata (pd.DataFrame): Metadata with file info.
        base_directory (str): Directory with input CSV files.
        output_directory (str): Directory for saving stacked files.

    Returns:
        pd.DataFrame: Combined DataFrame of all stacked groups.
    """
    os.makedirs(output_directory, exist_ok=True)  # Ensure output directory exists
    full_stacked_df = pd.DataFrame()
    log_file_path = os.path.join(output_directory, 'stack_log.txt')

    with open(log_file_path, 'w') as log_file:
        for segment_id, group in metadata.groupby('SegmentID'):
            if len(group) > 2:
                log_file.write(f"Skipping SegmentID '{segment_id}' due to more than 2 files.\n")
                continue
            
            stacked_data = pd.DataFrame()
            for i, row in group.iterrows():
                try:
                    file_path = os.path.join(base_directory, row['FileName'])
                    data = pd.read_csv(file_path)
                    data['SourceFile'] = row['FileName']
                    data['Speaker'] = 'A' if i == 0 else 'B'
                    stacked_data = pd.concat([stacked_data, data], ignore_index=True)
                except Exception as e:
                    log_file.write(f"Error processing {row['FileName']} in SegmentID '{segment_id}': {e}\n")
                    continue

            if not stacked_data.empty:
                stacked_data = process_turns_and_overlaps(stacked_data)
                save_segment_stacked_data(stacked_data, segment_id, output_directory, log_file)
                full_stacked_df = pd.concat([full_stacked_df, stacked_data], ignore_index=True)
    return full_stacked_df

def process_turns_and_overlaps(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'Turn', 'Backchannel', 'Overlap', and 'Contest' columns to a DataFrame of stacked segment data.

    Args:
        data (pd.DataFrame): Stacked DataFrame of segment data.

    Returns:
        pd.DataFrame: Updated DataFrame with turn, overlap, backchannel, and contest annotations.
    """
    data = data.sort_values(by='Start Time').reset_index(drop=True)
    data['Backchannel'] = 0
    data['Overlap'] = 0
    data['Turn'] = 0
    current_turn = 0
    prev_speaker = None
    prev_end_time = 0

    for i, row in data.iterrows():
        is_overlap = row['Start Time'] < prev_end_time and row['Speaker'] != prev_speaker
        if i > 0 and data.at[i - 1, 'Speaker'] != row['Speaker']:
            current_turn += 1
        data.at[i, 'Turn'] = current_turn if not is_overlap else f"Overlap_{current_turn}"
        data.at[i, 'Overlap'] = int(is_overlap)
        prev_speaker = row['Speaker']
        prev_end_time = row['End Time']

    data['Contest'] = 'uncontested'
    for i in range(len(data)):
        if data.at[i, 'Overlap'] == 1:
            neighborhood = data['Overlap'].iloc[max(i-5, 0):min(i+6, len(data))]
            data.at[i, 'Contest'] = 'contested' if neighborhood.sum() > 1 else 'uncontested'
    return data

def save_segment_stacked_data(data: pd.DataFrame, segment_id: str, output_dir: str, log_file):
    """Save each segment's stacked data to a CSV file and log the action."""
    try:
        output_path = os.path.join(output_dir, f'stacked_{segment_id}.csv')
        data.to_csv(output_path, index=False)
        logging.info("Saved stacked file for SegmentID %s at %s", segment_id, output_path)
    except Exception as e:
        log_file.write(f"Error saving stacked file for SegmentID '{segment_id}': {e}\n")

def main():
    base_directory = os.path.join('Output', 'super_May22', 'Text')
    output_directory = os.path.join('Output', 'super_May22', 'Segment_pairs')
    metadata_path = './Output/super_May22/files_metadata.csv'

    meta_df = load_metadata(metadata_path)
    full_stacked_df = stack_files_by_group(meta_df, base_directory, output_directory)

    full_output_path = os.path.join(output_directory, 'full_stacked_output.csv')
    # full_stacked_df.to_csv(full_output_path, index=False)
    logging.info("Saved full stacked DataFrame at %s", full_output_path)

if __name__ == "__main__":
    main()

        
# if __name__ == "__main__":

#     # Define the base directory where the CSV files are located (adjust this as needed)
#     base_directory = os.path.join('Output', 'super_May22', 'Text')

#     # Create the output directory called Segment_pairs in the base directory
#     output_directory = os.path.join('Output', 'super_May22', 'Segment_pairs')

#     # Use the function to merge the files, log groups with more than 2 files, and save the results
#     final_stacked_df = stack_files_by_group(meta_df, base_directory, output_directory)

#     # Output the merged data for each group
#     print(final_stacked_df)