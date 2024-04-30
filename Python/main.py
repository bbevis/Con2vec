import os
import pandas as pd
import datetime
import visual

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
            print(file_path)
            metadata = get_file_metadata(file_path, filename, dirpath)
            all_files_metadata.append(metadata)
            
    all_files_metadata = pd.DataFrame(all_files_metadata)
    all_files_metadata['group'] = all_files_metadata['group'].str.split('/').str[-1]
    all_files_metadata = all_files_metadata[~all_files_metadata['group'].str.contains('dem|Dem|initia|empiri|Data_super_icbs')]

    # Create a new binary column based on whether 'filename' contains 'audio'
    all_files_metadata['is_audio'] = all_files_metadata['filename'].str.contains('audio').astype(int)
    all_files_metadata['is_video'] = all_files_metadata['filename'].str.contains('video').astype(int)

    print("Metadata saved to files_metadata.csv")
    return all_files_metadata

# data_directory = 'Data_super_icbs/'  # Change this to the directory you want to start from
# files_metadata = get_all_files_with_metadata(data_directory)

# # Create a DataFrame from the metadata
# df = pd.DataFrame(files_metadata)
# df['group'] = df['group'].str.split('/').str[-1]


# df = df[~df['group'].str.contains('dem|Dem|initia|empiri|Data_super_icbs')]

# # Create a new binary column based on whether 'filename' contains 'audio'
# df['is_audio'] = df['filename'].str.contains('audio').astype(int)
# df['is_video'] = df['filename'].str.contains('video').astype(int)


# # Save the DataFrame to a CSV file
# df.to_csv('./Output/files_metadata.csv', index=False)

# print("Metadata saved to files_metadata.csv")


def extract_video_raw(data_dir, video_filenames):
    
    # data_dir is the folder that contains the raw file

    for dirpath, dirnames, filenames in os.walk(data_dir):
        print('HERE')
        print(dirpath)
        print(dirnames)
        print(filenames)
        for filename in filenames:
            if filename in video_filenames:
                file_path = os.path.join(dirpath, filename)
                vf = visual.visual_features(file_path,
                            'Pretrained_models/face_landmarker_v2_with_blendshapes.task',
                            'Pretrained_models/gesture_recognizer.task')
                resAll = vf.raw_outputs()
                resAll.to_csv('Output/super_icbs/Video/' + filename + '.csv', index=False)

def video_files(size_threshold, below = True):
    
    # get list of video files only within a datasize
    if below == True:
        return metadata['filename'][(metadata['is_video'] == 1) & (metadata['file_size'] <= size_threshold)].to_list()
    elif below == False:
        return metadata['filename'][(metadata['is_video'] == 1) & (metadata['file_size'] >= size_threshold)].to_list()

data_directory = 'Data/super_icbs/'  # Change this to the directory you want to start from
metadata = get_all_files_with_metadata(data_directory)
# Save the DataFrame to a CSV file
metadata.to_csv('./Output/super_icbs/files_metadata.csv', index=False)

video_filenames = video_files(10)
print(video_filenames)
extract_video_raw(data_directory, video_filenames)
