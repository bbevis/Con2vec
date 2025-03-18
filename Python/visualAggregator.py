import pandas as pd
import os
from datetime import datetime

# Function to log progress messages
def log_message(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")

# Load the turnID data
turn_id_file = os.path.join('Output', 'super_May22', 'TurnIDs.csv')
turn_data = pd.read_csv(turn_id_file)  # Columns: PairID, PersonID, Turn Start, Turn End, Pair_Speaker_turn

# Load the lookup table for audio → video filename mapping
lut_file = os.path.join('Output', 'super_May22', 'video_audio_LUT.csv')
lut_data = pd.read_csv(lut_file)  # Columns: audio_filename, video_filename

# Create a dictionary for quick lookup
audio_to_video = dict(zip(lut_data['audio_filename'], lut_data['video_filename']))

# Initialize a list to store aggregated results
aggregated_results = []
missing_files = []
filtered_out_rows = []

# Get the total number of rows for progress reporting
total_turns = len(turn_data)

# Start processing
log_message(f"Starting processing of {total_turns} rows from turnID data.")

# Iterate over each row in the turn data
for index, row in turn_data.iterrows():
    folder_name = row['PairID']
    audio_filename = row['PersonID']
    start_time = row['Turn Start']
    end_time = row['Turn End']
    turn_id = row['Pair_Speaker_turn']

    # Use lookup table to get the corresponding video filename
    if audio_filename not in audio_to_video:
        log_message(f"WARNING: No matching video file for audio {audio_filename} (Row {index + 1}/{total_turns})")
        continue

    video_filename = audio_to_video[audio_filename]

    # Ensure the filename ends with .csv
    if not video_filename.endswith('.csv'):
        video_filename += '.csv'

    video_file_path = os.path.join('Output', 'super_May22', 'Video', video_filename)

    # Check if the file exists
    if not os.path.exists(video_file_path):
        log_message(f"WARNING: File not found: {video_file_path} (Row {index + 1}/{total_turns})")
        missing_files.append(video_filename)
        continue

    # Load the visual data for the current file
    try:
        visual_data = pd.read_csv(video_file_path)

        if 'timestamp_ms' not in visual_data.columns:
            log_message(f"ERROR: 'timestamp_ms' column missing in {video_file_path}. Skipping.")
            continue

        # Convert microseconds to seconds
        visual_data['Time_s'] = visual_data['timestamp_ms'] / 1_000_000

        # Rescale Video Time to Match Audio Time Range
        video_min_time = visual_data['Time_s'].min()
        video_max_time = visual_data['Time_s'].max()

        audio_min_time = start_time  # First Turn Start in the audio file
        audio_max_time = end_time    # Last Turn End in the audio file

        if video_max_time > video_min_time:
            # Rescale video timestamps to fit within the corresponding audio time range
            visual_data['Time_s'] = audio_min_time + (
                (visual_data['Time_s'] - video_min_time) * (audio_max_time - audio_min_time) / (video_max_time - video_min_time)
            )

        # Debugging: Print first few rescaled timestamps
        print(f"Rescaled Time Conversion for {video_filename}:")
        print(visual_data[['timestamp_ms', 'Time_s']].head(10))

        # Dynamically extract visual feature column names (exclude timestamp_ms)
        visual_features = [col for col in visual_data.columns if col != 'timestamp_ms']

    except Exception as e:
        log_message(f"ERROR: Failed to process file: {video_file_path}. Error: {str(e)}")
        continue

    # Filter data based on Turn Start and Turn End
    filtered_data = visual_data[(visual_data['Time_s'] >= start_time) & (visual_data['Time_s'] < end_time)]

    # Check if filtering dropped rows
    if filtered_data.empty:
        print(f"⚠️ No matching data for {video_filename} in range {start_time} to {end_time}.")
        print(f"Available Time_s range after rescaling: {visual_data['Time_s'].min()} to {visual_data['Time_s'].max()}")
        filtered_out_rows.append({
            'audio_filename': audio_filename,
            'video_filename': video_filename,
            'Turn Start': start_time,
            'Turn End': end_time,
            'Available Min Time_s': visual_data['Time_s'].min(),
            'Available Max Time_s': visual_data['Time_s'].max()
        })
        continue  # Skip this row

    # Perform aggregation (mean of visual features)
    aggregated = {
        'Pair_Speaker_turn': turn_id,
        'PairID': folder_name,
        'PersonID': audio_filename,
        'VideoFilename': video_filename,
        'Turn Start': start_time,
        'Turn End': end_time,
    }

    # Compute means for each visual feature dynamically
    for feature in visual_features:
        aggregated[feature] = filtered_data[feature].mean()

    aggregated_results.append(aggregated)

    # Print progress report
    log_message(f"Completed row {index + 1}/{total_turns} ({((index + 1) / total_turns) * 100:.2f}%).")

# Convert aggregated results to a DataFrame
aggregated_df = pd.DataFrame(aggregated_results)

# Save results
output_file = os.path.join('Output', 'super_May22', 'Visual_agg.csv')
if aggregated_df.empty:
    log_message("ERROR: No data to save. Check time filtering and file paths.")
else:
    aggregated_df.to_csv(output_file, index=False)
    log_message(f"Aggregated visual data saved to {output_file}.")

# Save time filtering mismatches
if filtered_out_rows:
    pd.DataFrame(filtered_out_rows).to_csv("Output/filtered_out_rows.csv", index=False)
    log_message(f"Saved list of {len(filtered_out_rows)} rows dropped due to time filtering to Output/filtered_out_rows.csv")
