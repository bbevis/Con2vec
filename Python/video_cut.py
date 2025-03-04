import os
import subprocess
import pandas as pd

def process_webm(input_video):
    """Convert WebM video to MP4 with H.264 encoding."""
    if not input_video.endswith(".webm"): return input_video  # Skip if already an MP4
    output_video = f"{os.path.splitext(input_video)[0]}.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-i", input_video, "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", output_video
    ], check=True)
    return output_video

def process_audio(input_audio):
    """Convert Opus/WebM audio to AAC."""
    output_audio = os.path.splitext(input_audio)[0] + ".aac"
    subprocess.run([
        "ffmpeg", "-y", "-i", input_audio, "-c:a", "aac", "-b:a", "128k", output_audio
    ], check=True)
    return output_audio

def merge_and_segment(video, audio, start, end, output_file, log):
    """Merge H.264 video with AAC audio and segment the output in a single step."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"Merging and segmenting: {video} + {audio} -> {output_file}")
    result = subprocess.run([
        "ffmpeg", "-y", "-i", video, "-i", audio, "-ss", str(start), "-to", str(end),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-c:a", "aac", "-b:a", "128k", "-strict", "experimental",
        "-shortest", output_file
    ], stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    
    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        log.write(f"ERROR: Merge failed for {output_file} (empty or corrupt file)\n")
        log.write(f"FFmpeg error: {result.stderr}\n")
    else:
        print(f"SUCCESS: Created {output_file}")

def process_all_files(data_folder, turn_ids_file, lookup_file, output_folder, log_file):
    """Processes all video and audio files by looping through the folders and using lookup tables.
       Ensures that turns are processed in the original order they appear in TurnIDs.csv.
    """
    turn_ids = pd.read_csv(turn_ids_file)
    lookup = pd.read_csv(lookup_file)
    
    print(f"Processing {len(turn_ids)} rows from {turn_ids_file}")
    with open(log_file, "w") as log:
        for _, row in turn_ids.iterrows():
            pair_id = row["PairID"]  # Subfolder containing the files
            person_id = row["PersonID"]  # Audio filename
            speaker = row["Speaker"]
            start_time = row["Turn Start"]
            end_time = row["Turn End"]
            output_filename = os.path.join(output_folder, f"{row['Pair_Speaker_turn']}.mp4")
            
            subfolder_path = os.path.join(data_folder, pair_id)
            if not os.path.isdir(subfolder_path):
                log.write(f"ERROR: Subfolder missing: {subfolder_path}\n")
                print(f"ERROR: Subfolder missing: {subfolder_path}")
                continue
            
            audio_file = os.path.join(subfolder_path, person_id)  # No .webm extension assumed
            video_match = lookup.loc[lookup["audio_filename"] == person_id, "video_filename"]
            
            if video_match.empty:
                log.write(f"ERROR: No entry in lookup table for audio file: {person_id}\n")
                print(f"ERROR: No entry in lookup table for audio file: {person_id}")
                continue
            
            video_file = os.path.join(subfolder_path, video_match.values[0])
            
            if not os.path.exists(video_file):
                log.write(f"ERROR: Video file not found in subfolder: {video_file}\n")
                print(f"ERROR: Video file not found in subfolder: {video_file}")
                continue
            if not os.path.exists(audio_file):
                # Try adding .webm if not found
                alternative_audio_file = audio_file + ".webm"
                if os.path.exists(alternative_audio_file):
                    audio_file = alternative_audio_file
                else:
                    log.write(f"ERROR: Audio file not found in subfolder: {audio_file}\n")
                    print(f"ERROR: Audio file not found in subfolder: {audio_file}")
                    continue
            
            try:
                print(f"Processing video: {video_file}")
                print(f"Processing audio: {audio_file}")
                video_file = process_webm(video_file)
                audio_file = process_audio(audio_file)
                merge_and_segment(video_file, audio_file, start_time, end_time, output_filename, log)
            except Exception as e:
                log.write(f"ERROR processing {person_id}: {str(e)}\n")
                print(f"ERROR processing {person_id}: {str(e)}")
# Define paths
data_folder = "./Data_super_May22"
turn_ids_file = "./Output/super_May22/TurnIDs.csv"
lookup_file = "./Output/super_May22/video_audio_LUT.csv"
output_folder = "./Output/super_May22/VideoAudio/"
log_file = "processing_log.txt"

# Process all files
process_all_files(data_folder, turn_ids_file, lookup_file, output_folder, log_file)
