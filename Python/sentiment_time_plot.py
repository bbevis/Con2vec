import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy.interpolate import interp1d

def plot_sentiment_time_series(word_level_data):
    
    # Create a binary column for "Contest" (1 if contested, 0 if uncontested)
    word_level_data['is_contested'] = word_level_data['Contest'].apply(lambda x: 1 if x == 'contested' else 0)
    word_level_data['Duration'] = word_level_data['End Time'] - word_level_data['Start Time']

    # Group by 'Speaker' and 'Turn' and perform aggregations
    turn_speaker_data = turn_data.groupby(['Speaker', 'Turn']).agg(
        start_time=('Start Time', 'min'),
        end_time=('End Time', 'max'),
        avg_word_count=('word_count', 'mean'),
        avg_sentiment=('Sentiment', 'mean'),
        count_backchannels=('Backchannel', 'sum'),
        mean_overlaps=('Overlap', 'mean'),
        mean_contested=('is_contested', 'mean'),
        avg_duration=('Duration', 'mean')
    ).reset_index()

    turn_speaker_data['midTime'] = (turn_speaker_data['start_time'] + turn_speaker_data['end_time']) / 2
    # turn_speaker_data['Turn_duration'] = turn_speaker_data['end_time'] - turn_speaker_data['start_time']
    turn_speaker_data['Turn_duration'] = (turn_speaker_data['start_time'] + turn_speaker_data['end_time']) / turn_speaker_data['midTime']
    turn_speaker_data = turn_speaker_data.sort_values(by='midTime')
    
    # Group by 'Turn' and perform aggregations
    turn_level_data = turn_data.groupby(['Turn']).agg(
        start_time=('Start Time', 'min'),
        end_time=('End Time', 'max'),
        avg_word_count=('word_count', 'mean'),
        avg_sentiment=('Sentiment', 'mean'),
        count_backchannels=('Backchannel', 'sum'),
        mean_overlaps=('Overlap', 'mean'),
        mean_contested=('is_contested', 'mean'),
        avg_duration=('Duration', 'mean')
    ).reset_index()

    turn_level_data['Time'] = (turn_level_data['start_time'] + turn_level_data['end_time']) / 2
    turn_level_data = turn_level_data.sort_values(by='Time')


    # Prepare the DataFrame by calculating midpoint times for each turn
    turn_speaker_data_A = turn_speaker_data[turn_speaker_data['Speaker'] == 'A']
    turn_speaker_data_B = turn_speaker_data[turn_speaker_data['Speaker'] == 'B']

    # Initialize a figure
    fig, (ax1, ax2) = plt.subplots(2, 1, layout='constrained', figsize=(12, 8))

    # Set a modern color palette for two speakers
    speaker_palette = ["#0e86d4", "#fd7f20"]

    # Increase font sizes for readability
    plt.rcParams.update({'font.size': 20})
    

    ax1.bar(
        turn_speaker_data_A['Turn'], 
        turn_speaker_data_A['avg_word_count'],
        align = 'center',
        width = 1,
        edgecolor='none',
        label="Speaker A",
        color=speaker_palette[0],
        alpha = .6
    )
    

    ax1.bar(
        turn_speaker_data_B['Turn'], 
        turn_speaker_data_B['avg_word_count'],
        align = 'center',
        width = 1,
        edgecolor='none',
        label="Speaker B",
        color=speaker_palette[1],
        alpha = .6
    )
    
    x_new = np.linspace(turn_level_data['Turn'].min(), turn_level_data['Turn'].max(),500)
    f = interp1d(turn_level_data['Turn'], turn_level_data['avg_sentiment'], kind='quadratic')
    y_smooth=f(x_new)
    
    ax2.plot(x_new,
             y_smooth,
             label="Sentiment",
             color='#4C1F7A',
             linewidth=2)
    
    ax3 = ax2.twinx()
    
    ax2.fill_between(turn_level_data['Turn'], turn_level_data['mean_contested'],
             label='Contested',
             alpha=0.3, step='mid', hatch='//', color = '#db1f48', facecolor = '#db1f48')
    
    mask_Back = turn_level_data['count_backchannels'] != 0
    
    ax2.scatter(turn_level_data['Turn'][mask_Back], turn_level_data['count_backchannels'][mask_Back] * 0,
                 label='Backchannel',
                 s = 100,
                 marker = 'x',
                 color='#007F73')
    
    mask_Over = turn_level_data['mean_overlaps'] != 0
    
    ax2.scatter(turn_level_data['Turn'][mask_Over], turn_level_data['mean_overlaps'][mask_Over] * 0,
                 label='Overlap',
                 marker = 'o',
                 s = 70,
                 facecolors='none',
                 color='#db1f48')
    
    
    # Customize the plot
    # fig.title('Conversational Dynamics by Speaker', pad=100)
    ax2.set_xlabel('Turn', fontsize = 20)
    ax1.set_ylabel('Word Count\n(log scale)', fontsize = 20)
    ax1.set_yscale('log')
    ax1.margins(x=0)
    ax2.set_ylabel('Sentiment', fontsize = 20)
    ax2.margins(x=0)
    ax3.set_ylabel('Overlapping \nSpeakers', fontsize = 20)
    ax3.set_yticklabels([])
    ax3.set_yticks([])
    ax3.margins(x=0)
    # ax2.set_ylim(-1.2, 1.2)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    fig.legend(loc='upper center',
            #    bbox_to_anchor=(.5, 1.05),
               ncol=3,
               frameon=False)
    
    # Remove grid lines for a cleaner look
    plt.grid(False)

    # Show the plot
    plt.tight_layout()
    fig.subplots_adjust(top=0.85)  # Adjusts top spacing to fit the legend
    
    # Directory to save the plots
    save_dir = "plots"
    os.makedirs(save_dir, exist_ok=True)

    
    # Save plot to the specified directory
    file_path = os.path.join('Output', 'super_May22', save_dir, "sample_plot_4.png")
    plt.savefig(file_path, format='png', dpi=300)  # Save as PNG with 300 dpi

    plt.close(fig)
    # plt.show()

# Example usage
# Assuming turn_data is your DataFrame
# file = 'stacked_20240521_1823_WBLMayTR126E.csv'
# file = 'stacked_20240521_1823_WBLMay8MQZR5.csv'
# file = 'stacked_20240522_1325_S3WBLME15F3C.csv'
file ='stacked_20240522_1325_S3WBLMRMZ83G.csv'
turn_data = pd.read_csv(os.path.join('Output', 'super_May22', 'Segment_pairs', file))

plot_sentiment_time_series(turn_data)