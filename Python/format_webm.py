import os
import subprocess
import glob
from multiprocessing import Pool, cpu_count

# Define the main directory where video files are stored
VIDEO_DIRECTORY = "Data_super_May22"

# Define the new output directory for MP4 files
OUTPUT_DIRECTORY = "Output/super_May22/Video/mp4s"

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

# Find all WebM files (videos without extensions but containing "-video-")
video_files = glob.glob(os.path.join(VIDEO_DIRECTORY, "**", "*-video-*"), recursive=True)

# Define processing parameters
FRAME_RATE = 30  # Adjust if needed
RESOLUTION = "1920x1080"  # Adjust if needed

def process_video(input_file):
    """Converts a WebM video (without extension) to MP4 and saves it in Output/super_May22/Video/mp4s."""
    
    # Get the filename without the path
    base_filename = os.path.basename(input_file)
    
    # Define the new output file path inside "Output/super_May22/Video/mp4s"
    output_file = os.path.join(OUTPUT_DIRECTORY, base_filename + ".mp4")

    # Check if the MP4 file already exists (avoid duplicate processing)
    if os.path.exists(output_file):
        print(f"✅ Skipping {input_file}, already processed.")
        return

    print(f"🔄 Converting {input_file} to MP4...")

    # Optimized FFmpeg command for fastest conversion
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",  # Overwrite existing file
        "-i", input_file,  # Input file (even without extension)
        "-vf", f"scale={RESOLUTION},fps={FRAME_RATE}",  # Fix resolution & frame rate
        "-c:v", "libx264",  # Use H.264 encoding
        "-preset", "ultrafast",  # Fastest encoding (speed over file size)
        "-crf", "23",  # Good balance of quality and speed
        "-c:a", "aac",  # Use AAC audio codec
        "-strict", "experimental",
        output_file
    ]

    # Run FFmpeg command
    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # Suppress FFmpeg output
        print(f"✅ Processed video saved as: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error processing {input_file}: {e}")

# ✅ Use Multi-Processing to Convert Multiple Videos at Once
if __name__ == "__main__":
    num_processes = min(6, cpu_count() - 2)  # Limit to 6 processes to avoid overloading CPU
    print(f"🚀 Using {num_processes} parallel processes for video conversion...")

    with Pool(processes=num_processes) as pool:
        pool.map(process_video, video_files)
    
    print("✅ All videos processed!")
