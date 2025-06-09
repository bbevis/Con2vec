import pandas as pd
import numpy as np
import os
import glob
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from functools import partial
import warnings
import traceback
from multiprocessing import Lock

# === CONFIGURATION ===

base_dir = "Output/super_May22"
vocal_dir = os.path.join(base_dir, "Vocal")
video_dir = os.path.join(base_dir, "Video")
segment_pairs_dir = os.path.join(base_dir, "Segment_pairs")
lut_path = os.path.join(base_dir, "video_audio_LUT.csv")

emotion_columns = [
    "emotion_Angry", "emotion_Disgust", "emotion_Fear", "emotion_Happy",
    "emotion_Sad", "emotion_Surprise", "emotion_Neutral"
]

exclude_columns = ['Time_s', 'person_id', 'audio_person_id', 'video_person_id']

# Global lock for safe logging
log_lock = Lock()
log_file_path = os.path.join(base_dir, "segment_pairs_errors.log")

# === STEP 1: VOCAL + VIDEO MERGE (parallelized) ===

def merge_single_pair(row_dict):
    """Merge vocal + video for one LUT row."""
    audio_filename = row_dict['audio_filename']
    video_filename = row_dict['video_filename']
    person_id = row_dict['person_id']
    
    if not audio_filename.lower().endswith('.csv'):
        audio_filename += '.csv'
    if not video_filename.lower().endswith('.csv'):
        video_filename += '.csv'
    
    audio_path = os.path.join(vocal_dir, audio_filename)
    video_path = os.path.join(video_dir, video_filename)
    
    if not os.path.exists(audio_path) or not os.path.exists(video_path):
        return None  # Skip if missing
    
    try:
        df_audio = pd.read_csv(audio_path)
        df_video = pd.read_csv(video_path)
    except Exception as e:
        print(f"⚠️ Error reading files for {person_id}: {e}")
        return None
    
    df_video['Time_s'] = df_video['timestamp_ms'] / 1_000.0
    
    df_audio = df_audio.sort_values('Time_s').reset_index(drop=True)
    df_video = df_video.sort_values('Time_s').reset_index(drop=True)
    
    df_merged = pd.merge_asof(
        df_audio,
        df_video,
        on='Time_s',
        direction='nearest',
        tolerance=0.05
    )
    
    if df_merged.empty:
        return None
    
    audio_person_id = os.path.splitext(audio_filename)[0]
    video_person_id = os.path.splitext(video_filename)[0]
    
    df_merged['person_id'] = person_id
    df_merged['audio_person_id'] = audio_person_id
    df_merged['video_person_id'] = video_person_id
    
    return df_merged

def load_and_merge_vocal_video_parallel(max_workers=12):
    """Load and merge vocal + video data in parallel."""
    print("=== Step 1: Loading and merging vocal + video data ===")
    
    lut = pd.read_csv(lut_path)
    rows = lut.to_dict('records')
    
    merged_dfs = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(merge_single_pair, rows), total=len(rows)))
    
    for df in results:
        if df is not None:
            merged_dfs.append(df)
    
    if merged_dfs:
        df_all_merged = pd.concat(merged_dfs, ignore_index=True)
        print(f"\n✅ Final vocal + video merged dataframe: {df_all_merged.shape}")
        return df_all_merged
    else:
        print("⚠️ No merged vocal+video data.")
        return None

# === STEP 2: SEGMENT_PAIRS PROCESSING (parallelized) ===

def aggregate_video_features_for_words(video_df, seg_df, columns_to_keep=None):
    """
    Aggregates video features per word (based on seg_df start/end times).
    Returns a DataFrame with one row per word.

    Parameters:
    - video_df: DataFrame with Time_s and video features (must be numeric)
    - seg_df: Segment_pairs DataFrame with Start Time and End Time columns
    - columns_to_keep: list of columns to aggregate (optional). If None, use all numeric columns.
    """
    import pandas as pd
    import numpy as np

    if video_df.empty:
        return pd.DataFrame(index=seg_df.index)

    video_df = video_df.copy()
    video_df = video_df.apply(pd.to_numeric, errors='coerce')

    if 'Time_s' not in video_df.columns:
        raise ValueError("Missing 'Time_s' in video data")

    if columns_to_keep is None:
        video_columns = [col for col in video_df.select_dtypes(include=[np.number]).columns if col not in ['Time_s', 'timestamp_ms']]
    else:
        video_columns = [col for col in columns_to_keep if col in video_df.columns]

    video_times = video_df['Time_s'].values
    video_values = video_df[video_columns].values

    agg_data = []
    for _, word_row in seg_df.iterrows():
        start, end = word_row['Start Time'], word_row['End Time']
        mask = (video_times >= start) & (video_times <= end)

        if np.sum(mask) == 0:
            means = [np.nan] * len(video_columns)
        else:
            means = np.nanmean(video_values[mask], axis=0)

        agg_data.append(dict(zip(video_columns, means)))

    return pd.DataFrame(agg_data, index=seg_df.index)

def process_single_segment_file(args):
    """Process one Segment_pairs file using pre-aggregated video features per speaker."""
    seg_file, df_audio_by_person, df_video_by_person, audio_to_video_id = args
    word_rows = []

    try:
        seg_df = pd.read_csv(seg_file)
        unique_person_ids = seg_df['PersonID'].unique()
        if len(unique_person_ids) != 2:
            return word_rows  # Skip if not exactly 2 speakers
        
        personA_audio_id, personB_audio_id = unique_person_ids
        personA_video_id = audio_to_video_id.get(personA_audio_id)
        personB_video_id = audio_to_video_id.get(personB_audio_id)
        
        if personA_video_id not in df_video_by_person or personB_video_id not in df_video_by_person:
            # Skip if we cannot find both video dataframes
            return word_rows
        
        # Add Pause column
        seg_df = seg_df.sort_values('Start Time').reset_index(drop=True)
        pause_values = []
        for idx in range(len(seg_df)):
            if idx < len(seg_df) - 1:
                next_start_time = seg_df.loc[idx + 1, 'Start Time']
                end_time = seg_df.loc[idx, 'End Time']
                pause = max(0, next_start_time - end_time)
            else:
                pause = 0
            pause_values.append(pause)
        seg_df['Pause'] = pause_values
        
        # Aggregate video features for both speakers
        df_videoA = df_video_by_person[personA_video_id]
        df_videoB = df_video_by_person[personB_video_id]
        
        video_agg_A = aggregate_video_features_for_words(df_videoA, seg_df)
        video_agg_B = aggregate_video_features_for_words(df_videoB, seg_df)
        
        # Process each word row
        for idx, row in seg_df.iterrows():
            word_row = row.to_dict()
            
            # Aggregate vocal features for the speaker
            speaker_id = row['PersonID']
            speaker_df = df_audio_by_person.get(speaker_id)
            
            if speaker_df is not None and 'Time_s' in speaker_df.columns:
                speaker_time = speaker_df['Time_s'].values
                speaker_values = speaker_df.drop(columns=exclude_columns).values
                speaker_columns = speaker_df.drop(columns=exclude_columns).columns
                
                mask = (speaker_time >= row['Start Time']) & (speaker_time <= row['End Time'])
                if mask.any():
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        speaker_means = np.nanmean(speaker_values[mask], axis=0)
                else:
                    speaker_means = np.array([np.nan] * len(speaker_columns))
                
                for i, col in enumerate(speaker_columns):
                    word_row[col] = speaker_means[i]
            
            # Add speaker and listener video features using .iloc
            if row['Speaker'] == 'A':
                video_speaker_feats = video_agg_A.iloc[idx]
                video_listener_feats = video_agg_B.iloc[idx]
            else:
                video_speaker_feats = video_agg_B.iloc[idx]
                video_listener_feats = video_agg_A.iloc[idx]
            
            # Speaker video features → no prefix
            for col, val in video_speaker_feats.items():
                word_row[col] = val
            
            # Listener *emotion only* video features → _listener suffix
            for col in emotion_columns:
                val = video_listener_feats.get(col, np.nan)
                word_row[col + '_listener'] = val
            
            # Add SourceFile
            word_row['SourceFile'] = os.path.basename(seg_file)
            
            # Add speaker/listener IDs
            word_row['person_id'] = speaker_id
            word_row['audio_person_id'] = speaker_id
            word_row['video_person_id'] = audio_to_video_id.get(speaker_id)
            
            word_rows.append(word_row)
    
    except Exception as e:
        # Log any error
        error_message = f"Error processing {seg_file}: {str(e)}\n{traceback.format_exc()}\n"
        with log_lock:
            with open(log_file_path, "a") as log_file:
                log_file.write(error_message)
        return []
    
    return word_rows

def process_segment_pairs_parallel(df_audio_by_person, df_video_by_person, audio_to_video_id, max_workers=12):
    """Process Segment_pairs files in parallel."""
    print("\n=== Step 2: Processing Segment_pairs files (simplified) ===")
    
    segment_files = glob.glob(os.path.join(segment_pairs_dir, "*.csv"))
    print(f"Found {len(segment_files)} Segment_pairs files.")
    
    # For debugging: process first 2 files only
    # segment_files = segment_files[:2]
    
    args = [
        (seg_file, df_audio_by_person, df_video_by_person, audio_to_video_id)
        for seg_file in segment_files
    ]
    
    all_word_rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_single_segment_file, args), total=len(segment_files)))
    
    for word_rows in results:
        all_word_rows.extend(word_rows)
    
    df_word_level = pd.DataFrame(all_word_rows)
    print(f"\n✅ Final word-level dataframe: {df_word_level.shape}")
    return df_word_level


def check_segment_pairs_error_log():
    """Helper to check how many Segment_pairs files failed."""
    log_file_path = os.path.join(base_dir, "segment_pairs_errors.log")
    
    if not os.path.exists(log_file_path):
        print("\n✅ No errors found in Segment_pairs processing.")
        return
    
    with open(log_file_path, "r") as log_file:
        log_contents = log_file.read()
    
    # Find all filenames logged
    error_lines = [line for line in log_contents.splitlines() if line.startswith("Error processing ")]
    failed_files = [line.replace("Error processing ", "").split(":")[0] for line in error_lines]
    
    print(f"\n⚠️ {len(failed_files)} Segment_pairs files failed during processing.")
    if failed_files:
        print("Failed files:")
        for f in failed_files:
            print(f" - {f}")


# === MAIN PIPELINE ===

def main():
    # Clear previous error log
    with open(log_file_path, "w") as f:
        f.write("")

    max_workers = min(20, os.cpu_count() - 2)
    print(f"\n=== Using {max_workers} cores for parallel processing ===")

    # Step 0: Load LUT and build audio_to_video_id mapping
    lut = pd.read_csv(lut_path)
    audio_to_video_id = {
        row['audio_filename'].replace('.csv', ''): row['video_filename'].replace('.csv', '')
        for _, row in lut.iterrows()
    }
    print(f"\n✅ Loaded LUT: {len(lut)} rows")
    print(f"✅ Built audio_to_video_id mapping for {len(audio_to_video_id)} audio_person_ids")

    # Step 1: Load and merge vocal + video
    df_all_merged = load_and_merge_vocal_video_parallel(max_workers=max_workers)
    if df_all_merged is None:
        print("❌ Aborting — no vocal+video data.")
        return
    
    
    # Step 1.5: Pre-partition df_all_merged
    print("\n=== Pre-partitioning df_all_merged by person ===")
    df_audio_by_person = {
        person_id: df_all_merged[df_all_merged['audio_person_id'] == person_id].sort_values('Time_s')
        for person_id in df_all_merged['audio_person_id'].unique()
    }
    df_video_by_person = {
        person_id: df_all_merged[df_all_merged['video_person_id'] == person_id].sort_values('Time_s')
        for person_id in df_all_merged['video_person_id'].unique()
    }
    print(f"✅ Partitioned {len(df_audio_by_person)} audio_person_ids and {len(df_video_by_person)} video_person_ids")

    # Step 2: Process Segment_pairs in parallel
    df_word_level = process_segment_pairs_parallel(
        df_audio_by_person,
        df_video_by_person,
        audio_to_video_id,
        max_workers=max_workers
    )

    # Step 3: Save final word-level result
    output_word_level_path = os.path.join(base_dir, "word_level_merged.csv")
    df_word_level.to_csv(output_word_level_path, index=False)
    print(f"\n✅ Saved word-level merged data to: {output_word_level_path}")

    # Step 4: Check error log
    check_segment_pairs_error_log()



# === RUN ===
if __name__ == "__main__":
    main()
    # check_segment_pairs_error_log()

