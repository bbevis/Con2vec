import os
import pandas as pd


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


turn_data = combined_data.groupby(['PairID','PersonID', 'Turn', 'Speaker', 'Speaker_original', 'Speaker_turn','Pair_Speaker_turn']).agg({
        'Word': lambda x: ' '.join(x),
        'Start Time': 'min',
        'End Time': 'max'
    }).reset_index()

turn_data = turn_data.sort_values(by=['PairID', 'Start Time'])

# Calculate boundaries
turn_data['Turn_Boundary'] = (turn_data['Start Time'].shift(1) + turn_data['End Time']) / 2
turn_data.loc[0, 'Turn_Boundary'] = turn_data.loc[0, 'Start Time']

turn_data.rename(columns={'Word':'Sent', 'Start Time':'Turn Start', 'End Time': 'Turn End'}, inplace=True)


# Select the required columns
selected_columns = ['Pair_Speaker_turn', "PairID",  "PersonID","Speaker", "Speaker_original",
                    "Turn", "Speaker_turn",  "Turn_Boundary", "Turn Start", "Turn End"]
turn_data = turn_data[selected_columns]

# print(combined_data[['PairID', 'PersonID', 'Speaker']])

# Remove duplicate rows based on the 'Speaker_turn' column
# unique_data = turn_data.drop_duplicates(subset="Pair_Person_turn")

# Save the result to a new CSV file
file_out = os.path.join('Output', 'super_May22', 'TurnIDs.csv')
turn_data.to_csv(file_out, index=False)

print(f"Processed data has been saved to {file_out}")

  
