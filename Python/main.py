import os
import pandas as pd
import datetime
import visualExtractor as visual
import vocalExtractor as vocal
import textExtractor as text
import time
import asyncio

def get_metadata(data_dir):
    
    all_files_metadata = []
    
    for dirpath, dirnames, filenames in os.walk(data_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            metadata = {}
            metadata['filename'] = filename
            metadata['group'] = dirpath
            dt = datetime.datetime.fromtimestamp(os.path.getctime(file_path))
            metadata['date_created'] = dt.strftime('%Y-%m-%d')
            metadata['file_size'] = os.path.getsize(file_path) /1000000
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

def get_file_list(lower_threshold, upper_threshold, datatype, outpath):
    
    # get list of audio files only within a datasize range
    if datatype == 'video': dt = 'is_video'
    elif datatype == 'vocal': dt = 'is_audio'
        
    fs = metadata['filename'][(metadata[dt] == 1) & 
                                (metadata['file_size'] <= upper_threshold) & 
                                (metadata['file_size'] >= lower_threshold) &
                                (metadata['issues'] == '')].to_list()

    
    # Remove from list files where features already extracted
    excluded_fs = os.listdir(outpath)
    excluded_fs = [f.replace('.csv', '') for f in excluded_fs]
    
    fs = [f for f in fs if f not in excluded_fs]
    
    return fs
                   
def extract_raw_features(data_dir, data_type, type_filenames, outpath):
    
    # data_dir is the folder that contains the raw data folders and files

    for dirpath, dirnames, filenames in os.walk(data_dir):        
        for filename in filenames:
            if filename in type_filenames:
                
                try:
                    file_path = os.path.join(dirpath, filename)
                    
                    if data_type == 'video':
                        
                        vf = visual.visual_features(file_path,
                                    'Pretrained_models/face_landmarker_v2_with_blendshapes.task',
                                    'Pretrained_models/gesture_recognizer.task')
                        resAll = vf.raw_outputs()

                    elif data_type == 'vocal':
                        
                        vf = vocal.vocal_features(file_path)
                        resAll = vf.raw_outputs()
                        
                    elif data_type == 'text':
                        
                        tf = text.text_features(file_path)
                        resAll = asyncio.run(tf.transcribe_single_file(file_path))
                        
                    
                        
                    resAll.to_csv(outpath + filename + '.csv', index=False)
                    print(f"{filename} has been saved successfully.")
                    
                except Exception as e:
                    
                    error_message = f"Error: {str(e)} while processing {filename}"
                    print(error_message)
                    
                    # open metafile and record error message
                    metadata = pd.read_csv('./Output/super_May22/files_metadata.csv')
                    metadata['issues'].loc[metadata['filename'] == filename] = error_message
                    metadata.to_csv('./Output/super_May22/files_metadata.csv', index=False)
                    continue  # Continue to the next iteration
                

if __name__ == '__main__':
    
    start_time = time.time()
    print(os.getcwd())
    
    ############ meta data ############################
    data_directory = 'Data_super_May22/'  # Change this to the directory you want to start from
    metadata = get_metadata(data_directory)
    # Save the DataFrame to a CSV file
    # metadata.to_csv('./Output/super_May22/files_metadata.csv', index=False)

    ############ video features ############################
    # video_filenames = get_file_list(1, 800, 'video','Output/super_May22/Video/')
    # print(video_filenames)

    # extract_raw_features(data_directory, 'video', video_filenames, 'Output/super_May22/Video/')
    
    ############ vocal features ############################
    # vocal_filenames = get_file_list(0, 10, 'vocal','Output/super_May22/Vocal/')
    # print(vocal_filenames)

    # extract_raw_features(data_directory, 'vocal', vocal_filenames, 'Output/super_May22/Vocal/')
    
    ############ text features ############################
    # text_filenames = get_file_list(0, 10, 'vocal','Output/super_May22/Text/')
    # print(text_filenames)

    # extract_raw_features(data_directory, 'text', text_filenames, 'Output/super_May22/Text/')
    
    ############ run time ############################
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"The code took {elapsed_time:.2f} minutes to run.")
    
    
