import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm

def plot_sentiment_time_series(turn_data):
    
    # Create a binary column for "Contest" (1 if contested, 0 if uncontested)
    turn_data['is_contested'] = turn_data['Contest'].apply(lambda x: 1 if x == 'contested' else 0)
    
    turn_data['Duration'] = turn_data['End Time'] - turn_data['Start Time']

    # Group by 'Speaker' and 'Turn' and perform aggregations
    turn_data = turn_data.groupby(['Speaker', 'Turn']).agg(
        start_time=('Start Time', 'min'),
        end_time=('End Time', 'max'),
        avg_word_count=('word_count', 'mean'),
        avg_sentiment=('Sentiment', 'mean'),
        count_backchannels=('Backchannel', 'sum'),
        count_overlaps=('Overlap', 'sum'),
        prop_contested=('is_contested', 'mean'),
        avg_duration=('Duration', 'mean')
    ).reset_index()

    
    turn_data['Time'] = (turn_data['start_time'] + turn_data['end_time']) / 2
    
    turn_data = turn_data.sort_values(by='Time')


    # Prepare the DataFrame by calculating midpoint times for each turn
    turn_data_A = turn_data[turn_data['Speaker'] == 'A']
    turn_data_B = turn_data[turn_data['Speaker'] == 'B']
    

    # Initialize a figure
    fig, (ax1, ax2) = plt.subplots(2, 1, layout='constrained', figsize=(12, 8))

    # Set a modern color palette for two speakers
    speaker_palette = ["#0e86d4", "#fd7f20"]

    # Increase font sizes for readability
    plt.rcParams.update({'font.size': 20})
    

    ax1.bar(
        turn_data_A['Turn'], 
        turn_data_A['avg_word_count'],
        align = 'center',
        width = 1,
        edgecolor='none',
        label="Speaker A",
        color=speaker_palette[0],
        alpha = .6
    )
    

    ax1.bar(
        turn_data_B['Turn'], 
        turn_data_B['avg_word_count'],
        align = 'center',
        width = 1,
        edgecolor='none',
        label="Speaker B",
        color=speaker_palette[1],
        alpha = .6
    )
    
       
    
    ax2.plot(turn_data['Turn'],
                turn_data['avg_sentiment'],
                label="Sentiment",
                color='#4C1F7A')
    
    ax2.fill_between(turn_data['Turn'], turn_data['prop_contested'],
             label='Contested',
             alpha=0.3, step='mid', hatch='//', color = '#db1f48', facecolor = '#db1f48')
    
    mask_Back = turn_data['count_backchannels'] != 0
    
    ax2.scatter(turn_data['Turn'][mask_Back], turn_data['count_backchannels'][mask_Back] * 0,
                 label='Backchannel',
                 s = 70,
                 marker = 'x',
                 color='#007F73')
    
    mask_Over = turn_data['count_overlaps'] != 0
    
    ax2.scatter(turn_data['Turn'][mask_Over], turn_data['count_overlaps'][mask_Over] * 0,
                 label='Overlap',
                 marker = 'o',
                 s = 50,
                 facecolors='none',
                 color='#db1f48')
    
    
    # Customize the plot
    # fig.title('Conversational Dynamics by Speaker', pad=100)
    ax2.set_xlabel('Turn', fontsize = 20)
    ax1.set_ylabel('Word Count', fontsize = 20)
    ax2.set_ylabel('Sentiment & \n Overlapping Speakers', fontsize = 20)
    ax2.set_ylim(-1.2, 1.2)
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