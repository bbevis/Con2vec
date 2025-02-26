import os
import pandas as pd
import numpy as np

df = pd.read_csv('./Output/super_May22/files_metadata.csv')

# Function to extract unique ID, video, and audio filenames
def extract_info(filename, df):
    person_id = filename.split("-cam")[0]  # Extract unique ID
    if "video" in filename:
        df.loc[df["filename"].str.startswith(person_id), "video_filename"] = filename
    elif "audio" in filename:
        df.loc[df["filename"].str.startswith(person_id), "audio_filename"] = filename
    return person_id

# Apply the function
df["person_id"] = df["filename"].apply(lambda x: extract_info(x, df))

# Keep only relevant columns
df = df[["person_id", "audio_filename", "video_filename"]].drop_duplicates()

df.to_csv('./Output/super_May22/video_audio_LUT.csv', index=False)