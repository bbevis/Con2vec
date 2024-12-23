
import os
import pandas as pd
from textblob import TextBlob

# Define the paths
folder_path = os.path.join('Output', 'super_May22', 'Segment_Pairs')

# Create an empty DataFrame to store the combined results
combined_data = pd.DataFrame()

# Loop through all files in Folder A
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):  # Ensure the file is a CSV
        file_path = os.path.join(folder_path, file_name)
        
        # Load the file
        df = pd.read_csv(file_path)
        
        # Append the data to the combined DataFrame
        combined_data = pd.concat([combined_data, df], ignore_index=True)
        
combined_data['Pair_Speaker_turn'] = combined_data['PairID'] + '_' + combined_data['Speaker'] + '_' + combined_data['Speaker_original'] + '_' + combined_data['Turn'].astype(str)

turn_data = combined_data.groupby(['Pair_Speaker_turn', 'PairID','PersonID', 'Speaker', 'Speaker_original', 'Turn']).agg({
        'Word': lambda x: ' '.join(x),
        'Start Time': 'min',
        'End Time': 'max',
        'Backchannel': 'mean',
        'Overlap': 'mean',
        'Contested': 'mean'
    }).reset_index()

turn_data['Duration'] = turn_data['End Time'] - turn_data['Start Time']

# Calculate sentiment for each turn
turn_data['Sentiment'] = turn_data['Word'].apply(lambda text: TextBlob(text).sentiment.polarity)

# Estimate word count
turn_data['word_count'] = turn_data['Word'].str.split().str.len()

turn_data = turn_data.sort_values(by=['PairID', 'Start Time'])

# Save the result to a new CSV file
file_out = os.path.join('Output', 'super_May22', 'Text_agg.csv')
turn_data.to_csv(file_out, index=False)

print(f"Processed data has been saved to {file_out}")