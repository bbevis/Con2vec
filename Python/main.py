import os
import pandas as pd
import datetime
import visualExtractor as visual

def get_file_metadata(file_path, filename, dirpath):
    
    metadata = {}
    metadata['filename'] = filename
    metadata['group'] = dirpath
    dt = datetime.datetime.fromtimestamp(os.path.getctime(file_path))
    metadata['date_created'] = dt.strftime('%Y-%m-%d')
    metadata['file_size'] = os.path.getsize(file_path) /1000000

    return metadata

def get_all_files_with_metadata(data_dir):
    
    all_files_metadata = []
    
    for dirpath, dirnames, filenames in os.walk(data_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            metadata = get_file_metadata(file_path, filename, dirpath)
            all_files_metadata.append(metadata)
            
    all_files_metadata = pd.DataFrame(all_files_metadata)
    all_files_metadata['group'] = all_files_metadata['group'].str.split('/').str[-1]
    all_files_metadata = all_files_metadata[~all_files_metadata['group'].str.contains('dem|Dem|initia|empiri|Data_super_icbs')]

    # Create a new binary column based on whether 'filename' contains 'audio'
    all_files_metadata['is_audio'] = all_files_metadata['filename'].str.contains('audio').astype(int)
    all_files_metadata['is_video'] = all_files_metadata['filename'].str.contains('video').astype(int)
    
    all_files_metadata = all_files_metadata[all_files_metadata['filename'] != '.DS_Store']
    all_files_metadata['issues'] = ''

    print("Metadata saved to files_metadata.csv")
    return all_files_metadata

def extract_video_raw(data_dir, video_filenames):
    
    # data_dir is the folder that contains the raw data folders and files

    for dirpath, dirnames, filenames in os.walk(data_dir):
        
        print(dirpath)
        print(dirnames)
        print(filenames)
        
        for filename in filenames:
            if filename in video_filenames:
                
                try:
                    file_path = os.path.join(dirpath, filename)
                    vf = visual.visual_features(file_path,
                                'Pretrained_models/face_landmarker_v2_with_blendshapes.task',
                                'Pretrained_models/gesture_recognizer.task')
                    resAll = vf.raw_outputs()
                    resAll.to_csv('Output/super_May22/Video/' + filename + '.csv', index=False)
                    print(f"{filename} has been saved successfully.")
                    
                except Exception as e:
                    error_message = f"Error: {str(e)} while processing {filename}"
                    print(error_message)
                    
                    # open metafile and record error message
                    metadata = pd.read_csv('./Output/super_May22/files_metadata.csv')
                    metadata['issues'].loc[metadata['filename'] == filename] = error_message
                    metadata.to_csv('./Output/super_May22/files_metadata.csv', index=False)
                    continue  # Continue to the next iteration
                    
                    
def video_files(lower_threshold, upper_threshold):
    
    # get list of video files only within a datasize range
    fs = metadata['filename'][(metadata['is_video'] == 1) & 
                                (metadata['file_size'] <= upper_threshold) & 
                                (metadata['file_size'] >= lower_threshold)].to_list()
    
    # Remove from list videos where visual features already extracted
    excluded_fs = os.listdir('Output/super_May22/Video/')
    excluded_fs = [f.replace('.csv', '') for f in excluded_fs]
    fs = [f for f in fs if f not in excluded_fs]
    
    return fs


print(os.getcwd())
data_directory = 'Data_super_May22/'  # Change this to the directory you want to start from
metadata = get_all_files_with_metadata(data_directory)
# Save the DataFrame to a CSV file
metadata.to_csv('./Output/super_May22/files_metadata.csv', index=False)

video_filenames = video_files(50, 100)
print(video_filenames)

# extract_video_raw(data_directory, video_filenames)
