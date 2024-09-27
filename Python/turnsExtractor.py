import torch
import librosa
import numpy as np
import pandas as pd
import os
import time
from transformers import pipeline

# Function to classify a single audio segment
def classify_segment(segment, sr, segment_idx, segment_duration, classifier, threshold=0.5):
    try:
        # Ensure segment is a 1D numpy array (which it should be already from librosa)
        segment_np = np.array(segment)

        # Perform classification on the audio segment
        output = classifier(segment_np, sampling_rate=sr)

        # Get the most probable label and confidence score
        label = output[0]['label']
        score = output[0]['score']
        

        # Classify as "human" or "non-human" based on label and threshold
        is_human = label == "human_speech" and score >= threshold
        start_time = segment_idx * segment_duration
        end_time = (segment_idx + 1) * segment_duration
        return {
            'segment': segment_idx + 1,
            'start_time': start_time,
            'end_time': end_time,
            'is_human': is_human,
            'confidence': score
        }
    except Exception as e:
        print(f"Error classifying segment {segment_idx + 1}: {e}")
        return None

def classify_audio(input_file, classifier, threshold=0.5, segment_duration=1):
    try:
        # Load the audio file using librosa
        y, sr = librosa.load(input_file, sr=None)
    except Exception as e:
        print(f"Error loading audio file {input_file}: {e}")
        return []

    # Try to segment the audio
    try:
        segment_length = int(segment_duration * sr)
        segments = [y[i:i + segment_length] for i in range(0, len(y), segment_length)]
    except Exception as e:
        print(f"Error segmenting the audio: {e}")
        return []

    # Sequential processing of segments
    results = []
    for i, segment in enumerate(segments):
        result = classify_segment(segment, sr, i, segment_duration, classifier, threshold)
        if result:
            results.append(result)

    return results

def save_results_to_csv(results, input_file, output_csv, group):
    # Get filename and directory name
    filename = os.path.basename(input_file)
    # directory = os.path.dirname(input_file)

    # Add filename and directory columns
    for result in results:
        result['filename'] = filename
        result['group'] = group

    # Convert results to a DataFrame
    df = pd.DataFrame(results)

    # Save DataFrame to CSV
    df.to_csv(output_csv, index=False)


def turnExtractor(input_file, output_csv, group):
    # Main function to execute the classification and saving results
    # Load pre-trained model for audio classification (use GPU if available)
    try:
        device = 0 if torch.cuda.is_available() else -1
        classifier = pipeline(task="audio-classification", model="superb/hubert-large-superb-er", device=device)
    except Exception as e:
        print(f"Error loading the classification model: {e}")
        return

    # Classify the audio and get the results
    results = classify_audio(input_file, classifier)

    # Save results to a CSV file
    if results:
        save_results_to_csv(results, input_file, output_csv, group)
        print(f"Results saved to {output_csv}")
    else:
        print("No results to save due to errors.")
    return results
  
if __name__ == '__main__':
  
    start_time = time.time()
    
    dirpath = 'Data_super_May22'
    group = '20240522_1325_S3WBLM9W4J66'
    filename = '1716394969658-1a569948-8381-409b-9a04-fbb484b191a4-cam-audio-1716394970693'

    input_file = os.path.join(dirpath, group, filename)
    # output_csv = 'audio_classification_test.csv'
    output_csv = os.path.join('Output', 'super_May22', 'test_output_human_classifier.csv')
    turnExtractor(input_file, output_csv, group)
    
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"The code took {elapsed_time:.2f} minutes to run.")

