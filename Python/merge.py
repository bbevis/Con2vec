import os
import pandas as pd

# Define the paths
folder_path = os.path.join('Output', 'super_May22', 'Segment_Pairs')

output_file = os.path.join('Output', 'super_May22', 'Merged_data.csv')

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

# Save the final combined DataFrame to a CSV file
combined_data.to_csv(output_file, index=False)

print(f"Combining complete. Output saved to {output_file}.")