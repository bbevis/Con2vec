from googleapiclient.discovery import build
from google.oauth2 import service_account
import pandas as pd
import logging

# Configure logging for debugging and tracking execution steps
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Constants for file paths and Google Drive API settings
SERVICE_ACCOUNT_FILE = "./python/gcs_credentials.json"
SCOPES = ["https://www.googleapis.com/auth/drive"]
FOLDER_ID = "1H5YTtKfvnJu5ksI_JmkYPTMGshKNBLHO"
TEXT_AGG_FILE = "./Output/super_May22/text_agg.csv"
OUTPUT_CSV_FILE = "./Output/super_May22/video_urls.csv"

WORD_COUNT_THRESHOLD = 5  # Change to any desired threshold (e.g., 5, 8, etc.)

def authenticate_drive():
    """
    Authenticate with Google Drive API using a service account.
    
    Returns:
        drive_service: Authenticated Google Drive API service instance.
    """
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds)

def get_drive_files(drive_service, folder_id):
    """
    Fetch all video file metadata (name and ID) from a specified Google Drive folder.
    
    Args:
        drive_service: Authenticated Google Drive API service instance.
        folder_id: The ID of the Google Drive folder containing the video files.

    Returns:
        List of dictionaries containing file IDs and names.
    """
    files = []
    page_token = None

    while True:
        results = drive_service.files().list(
            q=f"'{folder_id}' in parents and mimeType != 'application/vnd.google-apps.folder'",
            fields="nextPageToken, files(id, name)",
            pageSize=1000,  # Fetch up to 1000 files per request
            pageToken=page_token
        ).execute()

        files.extend(results.get("files", []))
        page_token = results.get("nextPageToken")
        if not page_token:
            break  # Exit loop if no more pages are available

    logging.info(f"Total files fetched: {len(files)}")
    return files

def parse_filename(file_name):
    """
    Extract PairID, Turn Number, and Speaker from a given filename.
    
    Args:
        file_name: The name of the video file.

    Returns:
        pair_id: Unique identifier for the negotiation session.
        turn_number: The turn number in the negotiation.
        speaker: The speaker label (A or B).
    """
    parts = file_name.split("_")
    if len(parts) < 5:
        return None, None, None  # Skip files with unexpected naming format

    pair_id = "_".join(parts[:3])  # Extract PairID from the first three elements
    turn_number = int(parts[-1].replace(".mp4", ""))  # Extract numerical turn number
    speaker = parts[3]  # Extract speaker (A or B)

    return pair_id, turn_number, speaker

def load_text_agg(file_path):
    """
    Load text_agg.csv and create a lookup dictionary for word count.
    
    Args:
        file_path: Path to the text_agg.csv file.

    Returns:
        Dictionary with keys as (PairID, Turn) and values as word count.
    """
    text_df = pd.read_csv(file_path)

    # Create a lookup dictionary for fast word count retrieval
    return {
        (row["PairID"], row["Turn"]): row["word_count"]
        for _, row in text_df.iterrows()
    }

def filter_valid_sequences(video_sessions, turn_info, word_threshold):
    """
    Generate valid 5-turn sequences where each turn in the sequence has word_count > threshold.
    
    Args:
        video_sessions: Dictionary containing video sessions grouped by PairID.
        turn_info: Dictionary with word count information for each turn.
        word_threshold: The minimum word count required for a turn to be included.

    Returns:
        List of filtered sequences that meet the word count requirement.
    """
    csv_data = [["VideoSessionID", "VideoFile1", "VideoURL1", "VideoFile2", "VideoURL2",
                 "VideoFile3", "VideoURL3", "VideoFile4", "VideoURL4", "VideoFile5", "VideoURL5", "AssignedCount"]]

    session_counter = 1  # Unique session counter (V1, V2, V3, ...)

    for pair_id, videos in video_sessions.items():
        for i in range(len(videos) - 4):  # Ensure at least 5-turn sequences
            sequence = videos[i:i + 5]  # Extract 5 consecutive turns

            # Check if all turns in the sequence meet the word count requirement
            if all(turn_info.get((pair_id, turn_num), 0) > word_threshold for turn_num, _, _, _ in sequence):
                row = [f"V{session_counter}"]
                
                # Add video file names and URLs to the row
                for _, _, file_name, file_url in sequence:
                    row.extend([file_name, file_url])
                
                row.append(0)  # AssignedCount starts at 0
                csv_data.append(row)
                session_counter += 1  # Increment session counter for next valid sequence

    logging.info(f"Valid sequences generated: {len(csv_data) - 1}")
    return csv_data

def main():
    """
    Main execution function to:
    - Authenticate Google Drive API
    - Fetch video files from Google Drive
    - Parse filenames to extract session details
    - Load word count data from text_agg.csv
    - Filter sequences based on the configured word count threshold
    - Save the valid sequences to a CSV file
    """
    logging.info("Starting processing...")

    # Step 1: Authenticate & fetch Google Drive files
    drive_service = authenticate_drive()
    files = get_drive_files(drive_service, FOLDER_ID)

    # Step 2: Parse video filenames and organize them by session
    video_sessions = {}
    for file in files:
        file_name = file["name"]
        file_url = f"https://drive.google.com/file/d/{file['id']}/view"
        pair_id, turn_number, speaker = parse_filename(file_name)

        if pair_id is None:
            continue  # Skip invalid filenames

        if pair_id not in video_sessions:
            video_sessions[pair_id] = []
        video_sessions[pair_id].append((turn_number, speaker, file_name, file_url))

    # Step 3: Sort turns in each session by turn order
    for session in video_sessions:
        video_sessions[session].sort()  # Sort by turn number to ensure correct order

    logging.info(f"Processed negotiation sessions: {len(video_sessions)}")

    # Step 4: Load word count data from text_agg.csv
    turn_info = load_text_agg(TEXT_AGG_FILE)

    # Step 5: Filter sequences based on the configured word count threshold
    csv_data = filter_valid_sequences(video_sessions, turn_info, WORD_COUNT_THRESHOLD)

    # Step 6: Convert to DataFrame and save as CSV
    df = pd.DataFrame(csv_data[1:], columns=csv_data[0])  # Exclude header row
    df.to_csv(OUTPUT_CSV_FILE, index=False)

    logging.info(f"CSV file '{OUTPUT_CSV_FILE}' has been created successfully with {len(df)} rows.")

# Run the script if executed directly
if __name__ == "__main__":
    main()


