import pandas as pd
import os
from datetime import datetime


# Function to log progress messages
def log_message(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")
    
# Load the turnID data
turn_id_file = os.path.join('Output', 'super_May22', 'TurnIDs.csv')
turn_data = pd.read_csv(turn_id_file)  # Columns: folder_name, file_name, start_time, end_time, turn_id

# Initialize a list to store aggregated results
aggregated_results = []

# Get the total number of rows for progress reporting
total_turns = len(turn_data)

# Start processing
log_message(f"Starting processing of {total_turns} rows from turnID data.")

# Iterate over each row in the turn data
for index, row in turn_data.iterrows():
    folder_name = row['PairID']
    file_name = row['PersonID']
    file_name = file_name + str('.csv')
    start_time = row['Turn Start']
    end_time = row['Turn End']
    turn_id = row['Pair_Speaker_turn']
    
    # Construct the full path to the audio file
    audio_file_path = os.path.join('Output', 'super_May22', 'Vocal', file_name)
    
    # Check if the file exists
    if not os.path.exists(audio_file_path):
        log_message(f"WARNING: File not found: {audio_file_path} (Row {index + 1}/{total_turns})")
        continue
    
    # Load the audio data for the current file
    try:
        audio_data = pd.read_csv(audio_file_path)  # Columns: time, RMS, Pitch
        audio_data = audio_data.sort_values(by='Time_s')  # Ensure the data is sorted by time
    except Exception as e:
        log_message(f"ERROR: Failed to process file: {audio_file_path}. Error: {str(e)}")
        continue
    
    # Filter the audio data based on the start and end times
    filtered_data = audio_data[(audio_data['Time_s'] >= start_time) & (audio_data['Time_s'] < end_time)]
    
    # Perform aggregation (mean, sum, etc.)
    # Perform aggregation (mean of each feature)
    aggregated = {
        'Pair_Speaker_turn': turn_id,
        'PairID': folder_name,
        'PersonID': file_name,
        'Turn Start': start_time,
        'Turn End': end_time,
        'Pitch': filtered_data['Pitch'].mean(),
        'Loudness': filtered_data['Loudness'].mean(),
        'Pulse': filtered_data['Pulse'].mean(),
        'Speech_crispiness': filtered_data['Speech_crispiness'].mean(),
        'Speech_brightness': filtered_data['Speech_brightness'].mean(),
        'Frequency_spread': filtered_data['Frequency_spread'].mean(),
        'Speech_clarity': filtered_data['Speech_clarity'].mean(),
        'Tonal_complexity': filtered_data['Tonal_complexity'].mean(),
    }

    
    aggregated_results.append(aggregated)
    
    # Print progress report
    log_message(f"Completed row {index + 1}/{total_turns} ({((index + 1) / total_turns) * 100:.2f}%).")

# Convert aggregated results to a DataFrame
aggregated_df = pd.DataFrame(aggregated_results)

# Save the aggregated data to a new CSV file
output_file = os.path.join('Output', 'super_May22', 'Vocal_agg.csv')
try:
    aggregated_df.to_csv(output_file, index=False)
    log_message(f"Aggregated audio data has been successfully saved to {output_file}.")
except Exception as e:
    log_message(f"ERROR: Failed to save aggregated data to {output_file}. Error: {str(e)}")
    

