import pandas as pd
import numpy as np
import os
from textblob import TextBlob
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
        try:
            # Check if it starts before the previous word has ended (indicating overlap or backchannel)
            is_overlap = row['Start Time'] < prev_end_time and row['Speaker'] != prev_speaker

            # Classify as backchannel if surrounding speakers are different speakers,
            # and start_time_diff is > 10 showing infrequent interjection
            # and while next start time diff is < 5 showing that the previous speaker was still going
            if data.at[i, 'start_time_diff'] > 10 and \
                data.at[i+1, 'start_time_diff'] < 5 and \
                data.at[i, 'Speaker'] != (data.at[i-1, 'Speaker'] and data.at[i+1, 'Speaker']):
                    data.at[i, 'Backchannel'] = 1
                    data.at[i, 'Turn'] = current_turn
            # elif is_overlap and not is_backchannel_word:
            elif is_overlap:
                # Otherwise, classify as overlap if it's not a backchannel word
                data.at[i, 'Overlap'] = 1
                data.at[i, 'Turn'] = current_turn  # Keep the turn number the same
            elif i != 0 and (data.at[i - 1, 'Backchannel'] == 1 or data.at[i - 1, 'Overlap'] == 1):
                # print(i, row)
                data.at[i, 'Turn'] = current_turn
            else:
                # Increment turn only if it's a new speaker and not overlapping
                # or if the previous row was an overlap or backchannel
                if row['Speaker'] != prev_speaker:
                    current_turn += 1
                data.at[i, 'Turn'] = current_turn
                
            # Update previous speaker and end time trackers
            prev_speaker = row['Speaker']
            prev_end_time = row['End Time']
            
        except Exception as e:
            error_message = f"Error processing file: {str(e)}"
            logging.info(error_message)
            
            continue
    
    # Initialize the 'contest' column
    data['Contest'] = 'uncontested'

    # Iterate through the DataFrame
    for i in range(len(data)):
        if data['Overlap'][i] == 1 and \
            (np.sum(data['Overlap'][i-5:i]) == 0 and np.sum(data['Overlap'][i+1:i+5]) == 0):
                data.loc[i, 'Contest'] = 'uncontested'
        elif (data['Overlap'][i] == 1 and any(data['Overlap'][i+1:i+5] == 1)) or \
            (any(data['Overlap'][i-5:i+1] == 1) and any(data['Overlap'][i:i+5] == 1)):
            data.loc[i, 'Contest'] = 'contested'
            
    return data

def turn_level_outcomes(data: pd.DataFrame) -> pd.DataFrame:
    
    # Aggregate words by turn
    turn_data = data.groupby(['Turn', 'Speaker']).agg({
        'Word': lambda x: ' '.join(x),
        'Start Time': 'min',
        'End Time': 'max'
    }).reset_index()

    # Calculate sentiment for each turn
    turn_data['Sentiment'] = turn_data['Word'].apply(lambda text: TextBlob(text).sentiment.polarity)
    
    # Estimate word count
    turn_data['word_count'] = turn_data['Word'].str.split().str.len()

    # Calculate midpoint time boundaries between turns
    # turn_data['Time_Boundary'] = (turn_data['End Time'].shift(1) + turn_data['Start Time']) / 2
    # turn_data.loc[0, 'Time_Boundary'] = turn_data.loc[0, 'Start Time']  # Set the first boundary to its start time
    
    turn_data.rename(columns={'Word':'Sent', 'Start Time':'Turn Start', 'End Time': 'Turn End'}, inplace=True)
    
    return turn_data
    
def save_group_stacked_data(data: pd.DataFrame, group_id: str, output_dir: str, log_file):
    """Save each group's stacked data to a CSV file and log the action."""
    try:
        output_path = os.path.join(output_dir, f'stacked_{group_id}.csv')
        data.to_csv(output_path, index=False)
        logging.info("Saved stacked file for group %s at %s", group_id, output_path)
    except Exception as e:
        log_file.write(f"Error saving stacked file for group '{group_id}': {e}\n")

def stack_files_by_group(metadata, base_directory, output_directory):
    
    """
    Stacks files vertically for each group based on the metadata provided.
    Each file is stacked with an additional 'SourceFile' and 'Speaker' column.
    Logs the process and skips groups with more than 2 files.
    
    Args:
        metadata (pd.DataFrame): Metadata containing file information.
        base_directory (str): Path where the CSV files are located.
        output_directory (str): Directory to save the stacked files.
    
    Returns:
        pd.DataFrame: Final stacked DataFrame combining all groups.
    """
    
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
    # Define the log file path in the Segment_pairs directory
    log_file_path = os.path.join(output_directory, 'stack_log.txt')

    # Open the log file to write information
    with open(log_file_path, 'w') as log_file:

        # Loop through each unique group
        for group in metadata['group'].unique():
            # Filter the metadata for the current group
            group_files = metadata[metadata['group'] == group]['filename']

            # Check if there are exactly 2 files in the group
            if len(group_files) != 2:
                log_file.write(f"Group '{group}' contains {len(group_files)} files.\n")
                continue  # Skip this group if it doesn't have exactly 2 files

            # Initialize an empty list to store the stacked dataframes
            group_dfs = []

            # Loop through each file in the group and assign Speaker 'A' or 'B'
            for i, file in enumerate(group_files):
                file_path = os.path.join(base_directory, f"{file}.csv")
                try:
                    # Read the CSV file
                    df = pd.read_csv(file_path)

                    # Add a column for the file name and Speaker ('A' for the first file, 'B' for the second)
                    df['SourceFile'] = file
                    df['Speaker'] = 'A' if i == 0 else 'B'

                    # Append the dataframe to the list
                    group_dfs.append(df)
                except Exception as e:
                    log_file.write(f"Error reading {file} in group '{group}': {e}\n")
                    continue

            # Stack the dataframes vertically (i.e., concatenate them)
            if group_dfs:
                stacked_df = pd.concat(group_dfs, ignore_index=True)
                stacked_df = process_turns_and_overlaps(stacked_df)
                turn_df = turn_level_outcomes(stacked_df)
                merged_df = stacked_df.merge(turn_df, left_on = ['Turn', 'Speaker'], right_on = ['Turn', 'Speaker'])
                save_group_stacked_data(merged_df, group, output_directory, log_file)

    return merged_df

if __name__ == "__main__":
    base_directory = os.path.join('Output', 'super_May22', 'Text')
    output_directory = os.path.join('Output', 'super_May22', 'Segment_pairs')
    metadata_path = './Output/super_May22/files_metadata.csv'

    meta_df = load_metadata(metadata_path)
    stack_files_by_group(meta_df, base_directory, output_directory)

