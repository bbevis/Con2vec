import pandas as pd
import numpy as np
import os
import logging

# Configure logging to output to a file in the output directory
logging.basicConfig(filename='stacking_process.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

meta_df = pd.read_csv('./Output/super_May22/files_metadata.csv')
meta_df = meta_df[meta_df['is_audio'] == 1]


# Function to read and stack files based on SegmentID for each group and return the full stacked dataframe
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
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Define the log file path in the Segment_pairs directory
    log_file_path = os.path.join(output_directory, 'stack_log.txt')

    # Open the log file to write information
    with open(log_file_path, 'w') as log_file:
        # Initialize an empty list to store the stacked DataFrames from all groups
        all_group_dfs = []

        # Loop through each unique group
        for group in metadata['group'].unique():
            # Filter the metadata for the current group
            group_files = metadata[metadata['group'] == group]['filename']

            # Check if there are exactly 2 files in the group
            if len(group_files) > 2:
                log_file.write(f"Group '{group}' does not have exactly 2 files. Skipping...\n")
                continue  # Skip this group if it doesn't have exactly 2 files

            # Initialize an empty list to store the stacked dataframes
            group_dfs = []

            # Loop through each file in the group and assign Speaker 'A' or 'B'
            for i, file in enumerate(group_files):
                file_path = os.path.join(base_directory, file) + str('.csv')
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

            # Stack the dataframes vertically (i.e., concatenate them)
            if group_dfs:
                stacked_df = pd.concat(group_dfs, ignore_index=True)
                
                # Save the stacked DataFrame to the Segment_pairs directory
                output_file_path = os.path.join(output_directory, f'stacked_{group}.csv')
                stacked_df.to_csv(output_file_path, index=False)

                # Append the stacked DataFrame to the overall list
                all_group_dfs.append(stacked_df)

        # Concatenate all the DataFrames from all groups
        if all_group_dfs:
            final_df = pd.concat(all_group_dfs, ignore_index=True)
        else:
            final_df = pd.DataFrame()  # Return an empty DataFrame if no data was processed

    return final_df

# Define the base directory where the CSV files are located (adjust this as needed)
base_directory = os.path.join('Output', 'super_May22', 'Segments')

# Create the output directory called Segment_pairs in the base directory
output_directory = os.path.join('Output', 'super_May22', 'Segment_pairs')

# Use the function to merge the files, log groups with more than 2 files, and save the results
final_stacked_df = stack_files_by_group(meta_df, base_directory, output_directory)

# Output the merged data for each group
print(final_stacked_df)