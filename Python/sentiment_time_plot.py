import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_sentiment_time_series(turn_data):
    
    # Prepare the DataFrame by calculating midpoint times for each turn
    turn_data['Time'] = (turn_data['Start Time'] + turn_data['End Time']) / 2
    turn_data['Sentiment - A'] = np.where(turn_data['Speaker'] == 'A', turn_data['Sentiment'], 0)
    turn_data['Sentiment - B'] = np.where(turn_data['Speaker'] == 'B', turn_data['Sentiment'], 0)
    
    turn_data['Backchannel - A'] = np.where(turn_data['Speaker'] == 'A', turn_data['Backchannel'], 0)
    turn_data['Backchannel - B'] = np.where(turn_data['Speaker'] == 'B', turn_data['Backchannel'], 0)
    
    turn_data['Contest - A'] = np.where(turn_data['Speaker'] == 'A', turn_data['Contest'], 0)
    turn_data['Contest - B'] = np.where(turn_data['Speaker'] == 'B', turn_data['Contest'], 0)

    # Initialize a figure
    plt.figure(figsize=(14, 8))

    # Set a modern color palette for two speakers
    speaker_palette = ["#4C72B0", "#DD8452"]

    # Increase font sizes for readability
    plt.rcParams.update({'font.size': 14})
    
    plt.fill_between(turn_data['Time'], turn_data['Sentiment - A'],
                     label='Sentiment - A',
                         color=speaker_palette[0], alpha=0.5, step='mid')
    
    plt.fill_between(turn_data['Time'], turn_data['Sentiment - B'],
                     label='Sentiment - B',
                         color=speaker_palette[1], alpha=0.5, step='mid')
    
    
    # plt.scatter(turn_data['Time'], turn_data['Backchannel - A'] -.6,
    #              label='Backchannel - A',
    #              color=speaker_palette[0], marker='s', s=30)
    
    # plt.scatter(turn_data['Time'], turn_data['Backchannel - B'] -.6,
    #              label='Backchannel - B',
    #              color=speaker_palette[1], marker='s', s=30)
    

    # Customize the plot
    plt.title('Time Series of Sentiment and Conversational Dynamics by Speaker')
    plt.xlabel('Time')
    plt.ylabel('Sentiment and Conversational Dynamics')
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title='Legend')
    
    # Remove grid lines for a cleaner look
    plt.grid(False)

    # Show the plot
    plt.tight_layout()
    plt.show()

# Example usage
# Assuming turn_data is your DataFrame
file = 'stacked_20240521_1823_WBLMay6BBDES.csv'
turn_data = pd.read_csv(os.path.join('Output', 'super_May22', 'Segment_pairs', file))

plot_sentiment_time_series(turn_data)
