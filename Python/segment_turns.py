from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import logging

# Configure logging to output both to console and error log file
def setup_logging(output_folder):
    log_file = os.path.join(output_folder, 'error_log.txt')
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file, mode='a'),
                            logging.StreamHandler()
                        ])
    return log_file

# Function to find the optimal number of clusters using the elbow method for start_time_diff
def find_optimal_clusters_elbow_diff(data, max_k=10):
    silhouette_scores = []
    K_range = range(2, max_k + 1)
    for k in K_range:
        kmeans = KMeans(n_clusters=k)
        labels = kmeans.fit_predict(data)
        silhouette_scores.append(silhouette_score(data, labels))
    
    # Calculate the second derivative (difference of the difference) to find the "elbow"
    diffs = np.diff(silhouette_scores)
    elbow_k = np.argmin(diffs) + 2  # Adding 2 because the silhouette scores range starts at k=2

    # Plot the silhouette scores to visualize the elbow
    # plt.figure(figsize=(8, 5))
    # plt.plot(K_range, silhouette_scores, marker='o', linestyle='--')
    # plt.xlabel('Number of Clusters')
    # plt.ylabel('Silhouette Score')
    # plt.title('Silhouette Scores Elbow Method for start_time_diff')
    # plt.grid(True)
    # plt.show()

    return elbow_k

def process_file(file_path, output_folder):
    try: 
        logging.info(f'Processing file: {file_path}')
        
        data = pd.read_csv(file_path)
        
        # Step 1: Calculate the time difference between consecutive start times
        data['start_time_diff'] = data['Start Time'].diff().fillna(0)

        # Step 2: Apply elbow method to find the optimal number of clusters for start_time_diff
        start_time_diff = data['start_time_diff'].values.reshape(-1, 1)
        optimal_k_diff = find_optimal_clusters_elbow_diff(start_time_diff, max_k=4)
        
        # Step 3: Apply KMeans clustering with the optimal number of clusters
        # kmeans_diff_optimal = KMeans(n_clusters=optimal_k_diff)
        kmeans_diff_optimal = KMeans(n_clusters=2)
        data['ClusterID'] = kmeans_diff_optimal.fit_predict(start_time_diff)

        # Step 4: Use these boundaries to segment each word into different segments
        upper_bound_cluster = max(data['start_time_diff'][data['ClusterID'] == 0])
        lower_bound_next_cluster = min(data['start_time_diff'][data['ClusterID'] == 1])
        boundary_value = (upper_bound_cluster + lower_bound_next_cluster) / 2

        data['SegmentID'] = 1

        for i in range(1, len(data)):
            if data.loc[i, 'start_time_diff'] > boundary_value:
                data.loc[i, 'SegmentID'] = data.loc[i-1, 'SegmentID'] + 1
            else:
                data.loc[i, 'SegmentID'] = data.loc[i-1, 'SegmentID']

        # Save the output with SegmentID column
        output_file = os.path.join(output_folder, os.path.basename(file_path))
        data.to_csv(output_file, index=False)
        logging.info(f'File processed and saved as: {output_file}')
        
    except Exception as e:
        error_message = f"Error processing file {file_path}: {str(e)}"
        logging.error(error_message)

def process_all_files_in_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    log_file = setup_logging(output_folder)
    logging.info(f'Starting batch processing. Errors will be logged in: {log_file}')

    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_folder, filename)
            process_file(file_path, output_folder)

    logging.info('Batch processing complete.')
    
if __name__ == "__main__":
    input_folder = os.path.join('Output', 'super_May22', 'Text')  # Set your input folder path here
    output_folder = os.path.join('Output', 'super_May22', 'Segments')  # Set your output folder path here
    
    process_all_files_in_folder(input_folder, output_folder)