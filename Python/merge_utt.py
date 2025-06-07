import pandas as pd
import os
import glob

# === CONFIGURATION ===

# Define base directory and subdirectories
base_dir = "Output/super_May22"
vocal_dir = os.path.join(base_dir, "Vocal")
video_dir = os.path.join(base_dir, "Video")
segment_pairs_dir = os.path.join(base_dir, "Segment_pairs")
lut_path = os.path.join(base_dir, "video_audio_LUT.csv")

# Define which columns are emotion columns (from video data)
emotion_columns = [
    "emotion_Angry", "emotion_Disgust", "emotion_Fear", "emotion_Happy",
    "emotion_Sad", "emotion_Surprise", "emotion_Neutral"
]

# Columns to exclude when computing mean nonverbal features
# (we do not take the mean of Time_s or ID columns)
exclude_columns = ['Time_s', 'person_id', 'audio_person_id', 'video_person_id']

# === FUNCTIONS ===

def load_and_merge_vocal_video():
    """
    Load vocal and video data based on LUT, perform nearest merge,
    and return a single combined DataFrame df_all_merged.
    """
    print("=== Step 1: Loading and merging vocal + video data ===")
    
    lut = pd.read_csv(lut_path)
    merged_dfs = []

    for idx, row in lut.iterrows():
        audio_filename = row['audio_filename']
        video_filename = row['video_filename']
        
        # Ensure .csv extension is present
        if not audio_filename.lower().endswith('.csv'):
            audio_filename += '.csv'
        if not video_filename.lower().endswith('.csv'):
            video_filename += '.csv'
        
        person_id = row['person_id']
        
        # Build file paths
        audio_path = os.path.join(vocal_dir, audio_filename)
        video_path = os.path.join(video_dir, video_filename)
        
        # Check if both files exist
        if not os.path.exists(audio_path) or not os.path.exists(video_path):
            print(f"[{idx+1}/{len(lut)}] Skipping missing file(s): {audio_filename}, {video_filename}")
            continue
        
        # Load audio and video CSVs
        df_audio = pd.read_csv(audio_path)
        df_video = pd.read_csv(video_path)
        
        # Convert video timestamp from microseconds to seconds
        df_video['Time_s'] = df_video['timestamp_ms'] / 1_000_000.0
        
        # Sort dataframes by Time_s (required for merge_asof)
        df_audio = df_audio.sort_values('Time_s').reset_index(drop=True)
        df_video = df_video.sort_values('Time_s').reset_index(drop=True)
        
        # Nearest merge (within tolerance of 50 ms)
        df_merged = pd.merge_asof(
            df_audio,
            df_video,
            on='Time_s',
            direction='nearest',
            tolerance=0.05
        )
        
        # If no data matched, skip this pair
        if df_merged.empty:
            print(f"[{idx+1}/{len(lut)}] No matching rows for person {person_id}, skipping.")
            continue
        
        # Add identifying columns
        audio_person_id = os.path.splitext(audio_filename)[0]
        video_person_id = os.path.splitext(video_filename)[0]
        
        df_merged['person_id'] = person_id
        df_merged['audio_person_id'] = audio_person_id
        df_merged['video_person_id'] = video_person_id
        
        # Add merged df to list
        merged_dfs.append(df_merged)
        
        print(f"[{idx+1}/{len(lut)}] Merged {person_id}: {len(df_merged)} rows")
    
    # Combine all merged dataframes into one big df_all_merged
    if merged_dfs:
        df_all_merged = pd.concat(merged_dfs, ignore_index=True)
        print(f"\n✅ Final vocal + video merged dataframe: {df_all_merged.shape}")
        return df_all_merged
    else:
        print("⚠️ No merged vocal+video data.")
        return None

def process_segment_pairs(df_all_merged):
    """
    For each Segment_pairs file:
    - For each word row:
        - Compute mean speaker vocal+video features during word.
        - Compute mean listener emotion features during word.
        - Compute Pause duration.
        - During Pause, compute mean listener emotion features ONLY (no speaker features).
    Return a single word-level dataframe with all results.
    """
    print("\n=== Step 2: Processing Segment_pairs files ===")
    
    segment_files = glob.glob(os.path.join(segment_pairs_dir, "*.csv"))
    print(f"Found {len(segment_files)} Segment_pairs files.")
    
    all_word_rows = []
    
    for seg_file in segment_files:
        seg_df = pd.read_csv(seg_file)
        print(f"\nProcessing Segment_pairs file: {os.path.basename(seg_file)} ({len(seg_df)} rows)")
        
        # Detect unique PersonIDs in this conversation
        # NOTE: In Segment_pairs, PersonID refers to the audio PersonID (matches audio_person_id)
        # Speaker_A and Speaker_B are binary flags indicating who is speaking on each row
        unique_person_ids = seg_df['PersonID'].unique()
        if len(unique_person_ids) != 2:
            print(f"⚠️ Skipping file {os.path.basename(seg_file)} — found {len(unique_person_ids)} unique PersonIDs.")
            continue
        personA, personB = unique_person_ids
        print(f"Participants: {personA} (A), {personB} (B)")
        
        # Sort Segment_pairs by Start Time to enable pause calculation
        seg_df = seg_df.sort_values('Start Time').reset_index(drop=True)
        
        # Process each word row
        for idx, row in seg_df.iterrows():
            word_row = row.to_dict()  # Start with original word columns
            
            # === Determine speaker and listener IDs ===
            speaker_id = row['PersonID']
            listener_id = personA if speaker_id == personB else personB
            
            # === Define time window for this word ===
            start_time = row['Start Time']
            end_time = row['End Time']
            
            # === Aggregate speaker features during word ===
            speaker_rows = df_all_merged[
                (df_all_merged['audio_person_id'] == speaker_id) &
                (df_all_merged['Time_s'] >= start_time) &
                (df_all_merged['Time_s'] <= end_time)
            ]
            speaker_means = speaker_rows.drop(columns=exclude_columns).mean()
            for col in speaker_means.index:
                word_row[f'mean_speaker_{col}'] = speaker_means[col]
            
            # === Aggregate listener emotion features during word ===
            listener_rows = df_all_merged[
                (df_all_merged['video_person_id'] == listener_id) &
                (df_all_merged['Time_s'] >= start_time) &
                (df_all_merged['Time_s'] <= end_time)
            ]
            listener_emotions = listener_rows[emotion_columns].mean()
            for col in listener_emotions.index:
                word_row[f'mean_listener_emotion_{col}'] = listener_emotions[col]
            
            # === Pause calculation ===
            if idx < len(seg_df) - 1:
                next_start_time = seg_df.loc[idx + 1, 'Start Time']
                pause_duration = next_start_time - end_time
                if pause_duration > 0:
                    word_row['Pause'] = pause_duration
                    
                    # Listener emotion features during pause ONLY
                    listener_rows_pause = df_all_merged[
                        (df_all_merged['video_person_id'] == listener_id) &
                        (df_all_merged['Time_s'] > end_time) &
                        (df_all_merged['Time_s'] <= next_start_time)
                    ]
                    listener_emotions_pause = listener_rows_pause[emotion_columns].mean()
                    for col in listener_emotions_pause.index:
                        word_row[f'mean_listener_emotion_{col}_pause'] = listener_emotions_pause[col]
                else:
                    word_row['Pause'] = None
            else:
                word_row['Pause'] = None
            
            # Add source file for traceability
            word_row['SourceFile'] = os.path.basename(seg_file)
            
            # Add this word row to final list
            all_word_rows.append(word_row)
    
    # === FINAL DATAFRAME ===
    df_word_level = pd.DataFrame(all_word_rows)
    print(f"\n✅ Final word-level dataframe: {df_word_level.shape}")
    return df_word_level

def main():
    """
    Main pipeline:
    - Step 1: Load and merge vocal + video data.
    - Step 2: Process Segment_pairs and aggregate word-level data.
    - Save final result to word_level_merged.csv.
    """
    # Step 1
    df_all_merged = load_and_merge_vocal_video()
    if df_all_merged is None:
        print("❌ Aborting — no vocal+video data.")
        return
    
    # Step 2
    df_word_level = process_segment_pairs(df_all_merged)
    
    # Save result
    output_word_level_path = os.path.join(base_dir, "word_level_merged.csv")
    df_word_level.to_csv(output_word_level_path, index=False)
    print(f"\n✅ Saved word-level merged data to: {output_word_level_path}")

# === RUN ===
if __name__ == "__main__":
    main()
