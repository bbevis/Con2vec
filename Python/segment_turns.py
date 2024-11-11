import pandas as pd
from textblob import TextBlob
import os



def process_turn_data(base_directory, output_directory):
    
    
    # List all files in the folder
    for file in os.listdir(base_directory):
        # Check if the file is a CSV file
        if file.endswith('.csv'):
            try:
                file_path = os.path.join(base_directory, file)
                data = pd.read_csv(file_path)
                
                # Aggregate words by turn
                turn_data = data.groupby(['Turn', 'Speaker', 'SourceFile']).agg({
                    'Word': lambda x: ' '.join(x),
                    'Start Time': 'min',
                    'End Time': 'max'
                }).reset_index()
                
                # Calculate sentiment for each turn
                turn_data['Sentiment'] = turn_data['Word'].apply(lambda text: TextBlob(text).sentiment.polarity)
                
                # Calculate midpoint time boundaries between turns
                turn_data['Time_Boundary'] = (turn_data['End Time'].shift(1) + turn_data['Start Time']) / 2
                turn_data.loc[0, 'Time_Boundary'] = turn_data.loc[0, 'Start Time']  # Set the first boundary to its start time
                
                # Initialize columns for proportions
                turn_data['Prop_Backchannel_Time'] = 0.0
                turn_data['Prop_Overlap_Time'] = 0.0
                turn_data['Prop_Contested_Time'] = 0.0
                
                for idx, row in turn_data.iterrows():
                    turn_id = row['Turn']
                    turn_duration = row['End Time'] - row['Start Time']
                    
                    # Filter the data for the current turn
                    turn_segments = data[data['Turn'] == turn_id]

                    # Calculate proportions based on the presence of Backchannel, Overlap, and Contest
                    backchannel_duration = turn_segments[turn_segments['Backchannel'] == 1]['End Time'].sub(
                        turn_segments[turn_segments['Backchannel'] == 1]['Start Time']).sum()
                    overlap_duration = turn_segments[turn_segments['Overlap'] == 1]['End Time'].sub(
                        turn_segments[turn_segments['Overlap'] == 1]['Start Time']).sum()
                    contested_duration = turn_segments[turn_segments['Contest'] == 1]['End Time'].sub(
                        turn_segments[turn_segments['Contest'] == 1]['Start Time']).sum()

                    # Calculate proportion of each feature relative to the total turn duration
                    turn_data.at[idx, 'Prop_Backchannel_Time'] = backchannel_duration / turn_duration
                    turn_data.at[idx, 'Prop_Overlap_Time'] = overlap_duration / turn_duration
                    turn_data.at[idx, 'Prop_Contested_Time'] = contested_duration / turn_duration
                    
                    # Summary output for overall proportions across all turns
                    summary = {
                        "Overall_Prop_Backchannel_Time": turn_data['Prop_Backchannel_Time'].mean(),
                        "Overall_Prop_Overlap_Time": turn_data['Prop_Overlap_Time'].mean(),
                        "Overall_Prop_Contested_Time": turn_data['Prop_Contested_Time'].mean()
                    }
                    
                    # Define the output file path and save the processed data
                    output_file = os.path.join(output_directory, file)
                    turn_data.to_csv(output_file, index=False)

                    # Display processed turn-level data
                    print(f"Processed data saved to: {output_file}")
                
            except Exception as e:
                    
                error_message = f"Error: {str(e)} while processing {file}"
                print(error_message)
                    
                    
    return turn_data

if __name__ == '__main__':
    # Example usage
    base_directory = os.path.join('Output', 'super_May22', 'Segment_pairs')
    output_directory = os.path.join('Output', 'super_May22', 'Turns')
 

    turn_data = process_turn_data(base_directory, output_directory)

    
    # print(turn_data.head())









# # Configure logging to output both to console and error log file
# def setup_logging(output_folder):
#     log_file = os.path.join(output_folder, 'error_log.txt')
#     logging.basicConfig(level=logging.INFO, 
#                         format='%(asctime)s - %(levelname)s - %(message)s',
#                         handlers=[
#                             logging.FileHandler(log_file, mode='a'),
#                             logging.StreamHandler()
#                         ])
#     return log_file

# # Function to find the optimal number of clusters using the elbow method for start_time_diff
# def find_optimal_clusters_elbow_diff(data, max_k=10):
#     silhouette_scores = []
#     K_range = range(2, max_k + 1)
#     for k in K_range:
#         kmeans = KMeans(n_clusters=k)
#         labels = kmeans.fit_predict(data)
#         silhouette_scores.append(silhouette_score(data, labels))
    
#     # Calculate the second derivative (difference of the difference) to find the "elbow"
#     diffs = np.diff(silhouette_scores)
#     elbow_k = np.argmin(diffs) + 2  # Adding 2 because the silhouette scores range starts at k=2

#     # Plot the silhouette scores to visualize the elbow
#     # plt.figure(figsize=(8, 5))
#     # plt.plot(K_range, silhouette_scores, marker='o', linestyle='--')
#     # plt.xlabel('Number of Clusters')
#     # plt.ylabel('Silhouette Score')
#     # plt.title('Silhouette Scores Elbow Method for start_time_diff')
#     # plt.grid(True)
#     # plt.show()

#     return elbow_k

# def process_file(file_path, output_folder):
#     try: 
#         logging.info(f'Processing file: {file_path}')
        
#         data = pd.read_csv(file_path)
        
#         # Step 1: Calculate the time difference between consecutive start times
#         data['start_time_diff'] = data['Start Time'].diff().fillna(0)

#         # Step 2: Apply elbow method to find the optimal number of clusters for start_time_diff
#         start_time_diff = data['start_time_diff'].values.reshape(-1, 1)
#         optimal_k_diff = find_optimal_clusters_elbow_diff(start_time_diff, max_k=4)
        
#         # Step 3: Apply KMeans clustering with the optimal number of clusters
#         # kmeans_diff_optimal = KMeans(n_clusters=optimal_k_diff)
#         kmeans_diff_optimal = KMeans(n_clusters=2)
#         data['ClusterID'] = kmeans_diff_optimal.fit_predict(start_time_diff)

#         # Step 4: Use these boundaries to segment each word into different segments
#         upper_bound_cluster = max(data['start_time_diff'][data['ClusterID'] == 0])
#         lower_bound_next_cluster = min(data['start_time_diff'][data['ClusterID'] == 1])
#         boundary_value = (upper_bound_cluster + lower_bound_next_cluster) / 2

#         data['SegmentID'] = 1

#         for i in range(1, len(data)):
#             if data.loc[i, 'start_time_diff'] > boundary_value:
#                 data.loc[i, 'SegmentID'] = data.loc[i-1, 'SegmentID'] + 1
#             else:
#                 data.loc[i, 'SegmentID'] = data.loc[i-1, 'SegmentID']

#         # Save the output with SegmentID column
#         output_file = os.path.join(output_folder, os.path.basename(file_path))
#         data.to_csv(output_file, index=False)
#         logging.info(f'File processed and saved as: {output_file}')
        
#     except Exception as e:
#         error_message = f"Error processing file {file_path}: {str(e)}"
#         logging.error(error_message)

# def process_all_files_in_folder(input_folder, output_folder):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     log_file = setup_logging(output_folder)
#     logging.info(f'Starting batch processing. Errors will be logged in: {log_file}')

#     for filename in os.listdir(input_folder):
#         if filename.endswith('.csv'):
#             file_path = os.path.join(input_folder, filename)
#             process_file(file_path, output_folder)

#     logging.info('Batch processing complete.')
    
# if __name__ == "__main__":
#     input_folder = os.path.join('Output', 'super_May22', 'Text')  # Set your input folder path here
#     output_folder = os.path.join('Output', 'super_May22', 'Segments')  # Set your output folder path here
    
#     process_all_files_in_folder(input_folder, output_folder)