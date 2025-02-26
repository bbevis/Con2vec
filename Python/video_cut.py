import os
import subprocess

def process_webm(input_video):
    """Fix metadata and convert WebM to MP4 if needed."""
    if not input_video.endswith(".webm"): return input_video
    fixed_video, output_video = f"{input_video}_fixed.webm", f"{os.path.splitext(input_video)[0]}.mp4"
    print(f"Processing WebM: {input_video}...")
    subprocess.run(["ffmpeg", "-y", "-fflags", "+genpts", "-i", input_video, "-c", "copy", fixed_video], check=True)
    subprocess.run(["ffmpeg", "-y", "-i", fixed_video, "-c:v", "libx264", "-crf", "23", "-preset", "medium", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", output_video], check=True)
    return output_video

def process_opus_audio(input_audio):
    """Convert Opus audio to MP3 if needed."""
    if not input_audio.endswith(".webm"): return input_audio
    output_audio = os.path.splitext(input_audio)[0] + ".mp3"
    print(f"Converting Opus audio: {input_audio} to MP3...")
    subprocess.run(["ffmpeg", "-y", "-i", input_audio, "-c:a", "libmp3lame", "-b:a", "128k", output_audio], check=True)
    return output_audio

def segment_media(video, audio, start, end, output_folder):
    """Segment video and audio using ffmpeg."""
    os.makedirs(output_folder, exist_ok=True)
    video, audio = process_webm(video), process_opus_audio(audio)
    segmented_video_output = os.path.join(output_folder, "segmented_video.mp4")
    segmented_audio_output = os.path.join(output_folder, "segmented_audio.mp3")
    
    # Segment video
    subprocess.run(["ffmpeg", "-y", "-i", video, "-ss", str(start), "-to", str(end), "-c:v", "libx264", "-c:a", "aac", segmented_video_output], check=True)
    
    # Segment audio
    subprocess.run(["ffmpeg", "-y", "-i", audio, "-ss", str(start), "-to", str(end), "-c:a", "libmp3lame", "-b:a", "128k", segmented_audio_output], check=True)
    
    print(f"Segmented video saved to: {segmented_video_output}")
    print(f"Segmented audio saved to: {segmented_audio_output}")
    
# Manually specify file paths and segment times
video_file = "./Data_super_May22/20240516_1613_demoBSSRQ8/1715877775490-9d7fa1a0-b164-4d37-b4c8-4e970503b18a-cam-video-1715877776469"  # Replace with actual file path
audio_file = "./Data_super_May22/20240516_1613_demoBSSRQ8/1715877775490-9d7fa1a0-b164-4d37-b4c8-4e970503b18a-cam-audio-1715877776467"  # Replace with actual file path
start_time = 10  # Start time in seconds
end_time = 300  # End time in seconds
output_directory = "./Output/super_May22/VideoAudio"

print(f"Processing video: {video_file}")
print(f"Processing audio: {audio_file}")
print(f"Start time: {start_time}, End time: {end_time}")

segment_media(video_file, audio_file, start_time, end_time, output_directory)

