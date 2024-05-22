import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from deepface import DeepFace

class visual_features:
    
    def __init__(self, data, faceblend_model, handgesture_model):
        
        self.cap = cv2.VideoCapture(data)        
        self.BaseOptions = mp.tasks.BaseOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode        
        self.pt_blendshape = faceblend_model
        self.pt_handgesture = handgesture_model

    def face_cue_detector(self, mp_image):


        base_options = python.BaseOptions(model_asset_path=self.pt_blendshape)
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            output_face_blendshapes=True,
                                            output_facial_transformation_matrixes=True,
                                            running_mode=self.VisionRunningMode.IMAGE,
                                            num_faces = 1)

        
        detector = vision.FaceLandmarker.create_from_options(options)
        
        face_cue_detection_result = detector.detect(mp_image)
        
        # print('detection result: ', face_cue_detection_result)
        
        # print('face blendshape ', face_cue_detection_result.face_blendshapes)
        # print('length face blendshape: ', len(face_cue_detection_result.face_blendshapes))
        
        if len(face_cue_detection_result.face_blendshapes) > 0:
            face_blendshapes = face_cue_detection_result.face_blendshapes[0]
            names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
            scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
        else:
            names = ['no_face_detected']
            scores = ['']
        
        return names, scores


    def hand_gesture_detector(self, mp_image):

        options = vision.GestureRecognizerOptions(
            base_options=self.BaseOptions(model_asset_path=self.pt_handgesture),
            running_mode=self.VisionRunningMode.IMAGE)
        
        detector = vision.GestureRecognizer.create_from_options(options)
        
        gestures = detector.recognize(mp_image).gestures

        if len(gestures) > 0:
            gesture = gestures[0][0].category_name
        else:
            gesture = "No_hands"

        return gesture

    def emotion_detector(self, mp_image):
        
        emotions = DeepFace.analyze(mp_image, actions = ['emotion'], enforce_detection= False)
        
        scores = list(emotions[0]['emotion'].values())
        names = list(emotions[0]['emotion'].keys())
        
        return names, scores

    def raw_outputs(self):
        
        timestamps = []
        face_blendshapes_results = []
        hand_gesture_results = []
        emotion_results = []

        while True:

            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            timestamp_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            print('tiimestamp in seconds: ', timestamp_ms/1000)
            timestamps.append(timestamp_ms)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            
            hand_gestures = self.hand_gesture_detector(mp_image)
            hand_gesture_results.append(hand_gestures)
            
            face_blendshapes_names, face_blendshapes_scores = self.face_cue_detector(mp_image)
            print(face_blendshapes_scores)
            face_blendshapes_results.append(face_blendshapes_scores)

            emotion_name, emotion_scores = self.emotion_detector(frame)
            emotion_results.append(emotion_scores)
            
        handGestureRes = pd.get_dummies(pd.Series(hand_gesture_results)).astype(int)
        faceblendRes = pd.DataFrame(face_blendshapes_results, columns = face_blendshapes_names)
        emotionRes = pd.DataFrame(emotion_results, columns = emotion_name)

        resAll = pd.concat([faceblendRes, emotionRes, handGestureRes], axis = 1)
        # resAll['Timestamp_ms'] = timestamps
        resAll.insert(0, 'timestamp_ms', timestamps)
        self.cap.release()
        
        return resAll

if __name__ == '__main__':
    
    # video = moviepy.VideoFileClip('Data/1710526842273-c9be3e51-d151-47bd-a7fe-689236f35c0d-cam-video-1710526843251')
    # video.write_videofile('Data/test1.mp4')
    
    # /Users/bb320/Library/CloudStorage/GoogleDrive-burint@bnmanalytics.com/My Drive/Imperial/03_TeamofRivals/Con2vec/Data/super_icbs/20240312_1629_super_5KHZ83
    
    # base_path = '/Users/bb320/Library/CloudStorage/GoogleDrive-burint@bnmanalytics.com/My Drive/Imperial/03_TeamofRivals/Con2vec/'
    dirpath = 'Data_super_icbs'
    group = '20240312_1629_super_5KHZ83'
    filename = '1710326137265-4144e390-caf9-40c5-9424-9cc5f734cbb6-cam-video-1710326138273'
    filename_path =  os.path.join(dirpath, group, filename)

    # print("file exists?", os.path.exists(file_path))
    
    print('current directory: ', os.getcwd())
    print('filepath: ', filename_path)
    print("file exists?", os.path.exists(filename_path))

    vf = visual_features(filename_path,
                         'Pretrained_models/face_landmarker_v2_with_blendshapes.task',
                         'Pretrained_models/gesture_recognizer.task')
    
    
    resAll = vf.raw_outputs()
    resAll.to_csv('Output/super_icbs/test_output_super2.csv', index=False)