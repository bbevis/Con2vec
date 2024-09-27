from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt


filename = '1716394151038-9edee9b5-1b56-490c-b420-eb6f92a10f95-cam-audio-1716394152046.csv'
data_path = os.path.join('Output', 'super_May22', 'Text', filename)

# going to have to write our own!!!
# Pseudo code:
# for each word, calculate the distance between first and average of previoius cluster
# get cluster center
# if the average distance within the previous cluster to the cluster center is smaller than the
# distance between new cluster and cluster center, assign new cluster label to new word
# CHALLENGE - what happens if there are multiple single words?
# ANDWER: segment distance of words by only two clusters
# e.g. average length between clusters is 20 ms between turns and within sentence is .01 ms
# return only the optimal boundary threshold
# use the threshold to then iterate througheach word one at a time to label
# each word as part of previous segment if time diff is under threshold
# or new segment if greater than



# Extract the start time column
data = pd.read_csv(data_path)

# Function to find the optimal number of clusters using the elbow method for start_time_diff
def find_optimal_clusters_elbow_diff(data, max_k=50):
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
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, silhouette_scores, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores Elbow Method for start_time_diff')
    plt.grid(True)
    plt.show()

    return elbow_k


# Step 1: Calculate the time difference between consecutive start times
data['start_time_diff'] = data['Start Time'].diff().fillna(0)

# Step 2: Apply elbow method to find the optimal number of clusters for start_time_diff
start_time_diff = data['start_time_diff'].values.reshape(-1, 1)
optimal_k_diff = find_optimal_clusters_elbow_diff(start_time_diff, max_k=50)

# Step 3: Apply KMeans clustering with the optimal number of clusters
kmeans_diff_optimal = KMeans(n_clusters=optimal_k_diff)
data['ClusterID'] = kmeans_diff_optimal.fit_predict(start_time_diff)

# Identify the cluster boundaries for the start_time_diff
boundaries = []
for cluster_id in range(optimal_k_diff - 1):
    upper_bound_cluster = max(data['start_time_diff'][data['ClusterID'] == cluster_id])
    lower_bound_next_cluster = min(data['start_time_diff'][data['ClusterID'] == cluster_id + 1])
    boundary_value = (upper_bound_cluster + lower_bound_next_cluster) / 2
    boundaries.append(boundary_value)
    
print(boundaries)

# Step 4: Use these boundaries to segment each word into different segments
# THIS IS DEFINITELY WRONG. Definitely should not be based on the first boundary only....
data['SegmentID'] = (data['start_time_diff'] > boundaries[0]).cumsum()

# Group by SegmentID to find start and end times for each segment
# segments_boundary = data.groupby('SegmentID').agg(
#     Start_Segment=('Start Time', 'min'),
#     End_Segment=('End Time', 'max')
# ).reset_index()

# # Create a new CSV with the SegmentID and segment start and end times
output_path = os.path.join('Output', 'super_May22', 'Segments', filename)
data.to_csv(output_path, index=False)

