import pandas as pd
import numpy as np
import os
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


base_directory = os.path.join('Output', 'super_May22', 'Segment_pairs')
output_directory = os.path.join('Output', 'super_May22')
filename = 'stacked_20240521_1823_WBLMay1C89Q6.csv'
input_file_path = os.path.join(base_directory, filename)
output_file_path = os.path.join(output_directory, filename)
# print('filePath: ', file_path)

# Check if input file exists
if not os.path.isfile(input_file_path):
    logging.error(f"Input file '{input_file_path}' not found.")
    raise FileNotFoundError(f"Input file '{input_file_path}' not found.")

# Check if output directory exists, create if not
if not os.path.isdir(output_directory):
    logging.info(f"Output directory '{output_directory}' not found. Creating directory.")
    raise FileNotFoundError(f"Output directory '{output_directory}' not found.")


logging.info("Reading input data file.")
data = pd.read_csv(input_file_path)

# Define a set of common backchannel phrases
# backchannel_phrases = {"yes", "yeah", "okay", "uh-huh", "mm-hmm", "mh", "right"}

# Sort the data by 'Start Time' to ensure sequential analysis
data = data.sort_values(by='Start Time').reset_index(drop=True)

# Add columns for backchannel and overlap indicators, initialized to False
data['Backchannel'] = 0
data['Overlap'] = 0

# Initialize the 'Turns' column and set initial turn count and previous speaker
data['Turn'] = 0
current_turn = 0
prev_speaker = None
prev_end_time = 0  # Track end time for overlap detection


logging.info("Processing each row to assign turns, backchannel, and overlap indicators.")
# Process each row to determine turns, backchannel, and overlap
for i, row in data.iterrows():
    # Check if the word is a potential backchannel phrase
    # is_backchannel_word = row['Word'].lower() in backchannel_phrases
    
    # Check if it starts before the previous word has ended (indicating overlap or backchannel)
    is_overlap = row['Start Time'] < prev_end_time and row['Speaker'] != prev_speaker
    
    # Classify as backchannel if it's a backchannel word and overlaps with the previous speaker's speech
    # if is_overlap and is_backchannel_word:
    #     data.at[i, 'Backchannel'] = 1
    #     data.at[i, 'Turn'] = current_turn  # Keep the turn number the same
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
        data.at[i, 'Turn'] = f"Overlap_{current_turn}"  # Keep the turn number the same
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
  
# Initialize the 'contest' column
data['Contest'] = 'uncontested'

# Iterate through the DataFrame
for i in range(len(data)):
    # if (data['Overlap'][i] == 1):
    #     data.loc[i, 'Contest'] = 'contested'
    if data['Overlap'][i] == 1 and \
        (np.sum(data['Overlap'][i-5:i]) == 0 and np.sum(data['Overlap'][i+1:i+5]) == 0):
            data.loc[i, 'Contest'] = 'uncontested'
    elif (data['Overlap'][i] == 1 and any(data['Overlap'][i+1:i+5] == 1)) or \
        (any(data['Overlap'][i-5:i+1] == 1) and any(data['Overlap'][i:i+5] == 1)):
        data.loc[i, 'Contest'] = 'contested'
    

# Display the modified data with 'Turn' column
print(data[['Word', 'Start Time', 'End Time', 'Speaker', 'Turn']].head(20))

logging.info("Turn assignment complete. Saving the modified data to output file.")

data.to_csv(output_file_path, index=False)
logging.info("Data processing complete. File saved to output path.")
