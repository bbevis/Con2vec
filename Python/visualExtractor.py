import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
import subprocess
import logging
import time
import datetime
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from deepface import DeepFace
from multiprocessing import Pool, cpu_count
import traceback

############# GLOBAL SETTINGS #############
# Suppress Mediapipe warnings
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# Suppress TensorFlow & Mediapipe logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide TensorFlow warnings
os.environ['GLOG_minloglevel'] = '2'  # Hide Mediapipe internal logs

# Enable OpenCV multi-threading
cv2.setNumThreads(8)  # Use all 8 performance cores on M1 Pro

# Define the MP4 video folder (now separate from WebM)
MP4_VIDEO_DIRECTORY = "Output/super_May22/Video/mp4s"

# Define the output folder for CSV files
CSV_OUTPUT_DIRECTORY = "Output/super_May22/Video"
os.makedirs(CSV_OUTPUT_DIRECTORY, exist_ok=True)  # Ensure the folder exists


class visual_features:
    
    def __init__(self, data, faceblend_model, handgesture_model, enable_hand_gestures=False, enable_emotions=False):
        self.video_path = data  # Directly use the MP4 file
        
        # Enable/Disable Additional Features
        self.enable_hand_gestures = enable_hand_gestures
        self.enable_emotions = enable_emotions

        # Initialize OpenCV Video Capture with the MP4 file
        self.cap = cv2.VideoCapture(self.video_path)

        # Enable Apple M1 Hardware Acceleration for Faster Decoding
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"avc1"))  # Uses VideoToolbox on Mac

        # Explicitly set frame rate & resolution
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        self.pt_blendshape = faceblend_model
        self.pt_handgesture = handgesture_model
        self.face_blendshapes_names = []

        # Check actual video properties
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"✅ Video Properties | FPS: {self.actual_fps}, Resolution: {int(self.actual_width)}x{int(self.actual_height)}")

    def face_cue_detector(self, mp_image):
        base_options = python.BaseOptions(model_asset_path=self.pt_blendshape)
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                               output_face_blendshapes=True,
                                               output_facial_transformation_matrixes=True,
                                               running_mode=vision.RunningMode.IMAGE,
                                               num_faces=1)

        detector = vision.FaceLandmarker.create_from_options(options)
        face_cue_detection_result = detector.detect(mp_image)

        if len(face_cue_detection_result.face_blendshapes) > 0:
            face_blendshapes = face_cue_detection_result.face_blendshapes[0]
            names = [category.category_name for category in face_blendshapes]
            scores = [category.score for category in face_blendshapes]
        else:
            names, scores = ['no_face_detected'], ['']
        
        return names, scores

    def hand_gesture_detector(self, mp_image):
        """Detect hand gestures if enabled."""
        if not self.enable_hand_gestures:
            return "Hand gestures disabled"

        options = vision.GestureRecognizerOptions(
            base_options=python.BaseOptions(model_asset_path=self.pt_handgesture),
            running_mode=vision.RunningMode.IMAGE)
        
        detector = vision.GestureRecognizer.create_from_options(options)
        gestures = detector.recognize(mp_image).gestures

        return gestures[0][0].category_name if gestures else "No_hands"

    def emotion_detector(self, mp_image):
        """Detect facial emotions if enabled."""
        if not self.enable_emotions:
            return [], []

        emotions = DeepFace.analyze(mp_image, actions=['emotion'], enforce_detection=False)
        scores = list(emotions[0]['emotion'].values())
        names = list(emotions[0]['emotion'].keys())

        return names, scores

    def raw_outputs(self):
        """Extracts visual features from video frames."""
        timestamps = []
        face_blendshapes_results = []
        
        # Ensure face_blendshapes_names is always initialized
        if not self.face_blendshapes_names:
            self.face_blendshapes_names = [f"blendshape_{i}" for i in range(52)]

        frame_id = 0

        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                break  

            if frame_id % 2 != 0:  # Skip frames for speed
                frame_id += 1
                continue  

            timestamp_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamps.append(timestamp_ms)

            frame = cv2.resize(frame, (1920, 1080))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            if frame_id % 15 == 0:  # Face detection runs every .5 sec
                face_blendshapes_names, face_blendshapes_scores = self.face_cue_detector(mp_image)
                if len(face_blendshapes_names) == 52:  # Ensure correct naming
                    self.face_blendshapes_names = face_blendshapes_names  
            else:
                face_blendshapes_scores = [0] * 52  # Default zero values

            face_blendshapes_results.append(face_blendshapes_scores)
            frame_id += 1

        # Replace empty strings with NaN before converting to DataFrame
        face_blendshapes_results = [[np.nan if v == '' else v for v in row] for row in face_blendshapes_results]

        # Convert to DataFrame (now guaranteed to have numeric values)
        faceblendRes = pd.DataFrame(face_blendshapes_results, columns=self.face_blendshapes_names, dtype=np.float32)

        faceblendRes.insert(0, 'timestamp_ms', timestamps)
        
        # Remove rows where all extracted features are either NaN or zero (ignoring timestamp)
        mask = faceblendRes[self.face_blendshapes_names].isna().all(axis=1) | (faceblendRes[self.face_blendshapes_names] == 0).all(axis=1)
        faceblendRes = faceblendRes.loc[~mask]
        
        self.cap.release()
        return faceblendRes



### **🔹 Process All MP4 Files in Parallel**
def process_video(mp4_filename):
    """Processes an MP4 video, extracts features, and saves a CSV file."""
    start_time = time.time()
    
    # Define the expected CSV output file path
    csv_output_file = os.path.join(CSV_OUTPUT_DIRECTORY, os.path.basename(mp4_filename).replace(".mp4", ".csv"))

    # Skip processing if the CSV already exists
    if os.path.exists(csv_output_file):
        print(f"⏩ Skipping {mp4_filename}: CSV already exists at {csv_output_file}")
        return  
    
    print(f"🔄 Processing: {mp4_filename}")

    try:
        # Initialize visual feature extractor
        vf = visual_features(
            mp4_filename,
            'Pretrained_models/face_landmarker_v2_with_blendshapes.task',
            'Pretrained_models/gesture_recognizer.task'
        )

        # Extract features
        resAll = vf.raw_outputs()

        # Define CSV output path
        csv_output_file = os.path.join(CSV_OUTPUT_DIRECTORY, os.path.basename(mp4_filename).replace(".mp4", ".csv"))

        # Save extracted data
        resAll.to_csv(csv_output_file, index=False)
        print(f"✅ Successfully processed and saved: {csv_output_file}")

    except Exception as e:
        error_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_details = traceback.format_exc()

        error_message = (
            f"❌ ERROR at {error_time}\n"
            f"File: {mp4_filename}\n"
            f"Error Type: {type(e).__name__}\n"
            f"Error Message: {str(e)}\n"
            f"Traceback:\n{error_details}\n"
            "---------------------------------------------\n"
        )

        # Print error to console
        print(error_message)

        # Append error to log file
        with open("processing_errors.log", "a") as log_file:
            log_file.write(error_message)

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"✅ Completed: {mp4_filename} in {elapsed_time:.2f} min")


### ** Run Multi-Processing for Faster Execution**
if __name__ == '__main__':

    
    start_time = time.time()
    
    print("📂 Searching for MP4 files in:", MP4_VIDEO_DIRECTORY)
    
    # Collect all MP4 video file paths
    mp4_files = [os.path.join(MP4_VIDEO_DIRECTORY, f) for f in os.listdir(MP4_VIDEO_DIRECTORY) if f.endswith(".mp4")]

    # Check if there are files to process
    if not mp4_files:
        print("❌ No MP4 files found. Exiting.")
        exit()

    print(f"✅ Found {len(mp4_files)} MP4 files. Starting processing...")

    # Use multi-processing for efficiency (limits to 6 processes)
    with Pool(processes=6) as pool:
        pool.map(process_video, mp4_files)

    print("All videos processed successfully!")

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"⏳ Total execution time: {elapsed_time:.2f} minutes")
