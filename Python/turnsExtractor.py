import os
import pandas as pd

# Load the CSV file
# file_name = "Merged_data.csv"
file_in = os.path.join('Output', 'super_May22', 'Merged_data.csv')
data = pd.read_csv(file_in)

# Select the required columns
selected_columns = ['Pair_Speaker_turn', "PairID", "Speaker_turn", "PersonID", "Turn_Boundary", "Turn Start", "Turn End"]
data = data[selected_columns]

# Remove duplicate rows based on the 'Speaker_turn' column
unique_data = data.drop_duplicates(subset="Pair_Speaker_turn")

# Save the result to a new CSV file
file_out = os.path.join('Output', 'super_May22', 'Turn_boundaries.csv')
# output_file = "Turn_boundaries.csv"
unique_data.to_csv(file_out, index=False)

print(f"Processed data has been saved to {file_out}")

  
