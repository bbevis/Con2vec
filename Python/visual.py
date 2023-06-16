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
        # self.fps = cv2.VideoCapture(data).get(cv2.CAP_PROP_FPS)
        
        self.BaseOptions = mp.tasks.BaseOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        
        self.pt_blendshape = faceblend_model
        self.pt_handgesture = handgesture_model
        
        # cap = cv2.VideoCapture("Data/test_video.mov")
        # fps = cap.get(cv2.CAP_PROP_FPS)

        # BaseOptions = mp.tasks.BaseOptions
        # VisionRunningMode = mp.tasks.vision.RunningMode

    def face_cue_detector(self, mp_image):

        # FaceLandmarker = mp.tasks.vision.FaceLandmarker
        # FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

        base_options = python.BaseOptions(model_asset_path=self.pt_blendshape)
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            output_face_blendshapes=True,
                                            output_facial_transformation_matrixes=True,
                                            running_mode=self.VisionRunningMode.IMAGE,
                                            num_faces = 2)

        
        detector = vision.FaceLandmarker.create_from_options(options)
        
        face_cue_detection_result = detector.detect(mp_image)
        
        face_blendshapes = face_cue_detection_result.face_blendshapes[0]
        names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
        scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
        
        return names, scores


    def hand_gesture_detector(self, mp_image):

        GestureRecognizer = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions

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

    def collect_outputs(self):
        
        timestamps = []
        face_blendshapes_results = []
        hand_gesture_results = []
        emotion_results = []

        while True:

            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            timestamp_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamps.append(timestamp_ms)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            
            hand_gestures = self.hand_gesture_detector(mp_image)
            hand_gesture_results.append(hand_gestures)
            
            face_blendshapes_names, face_blendshapes_scores = self.face_cue_detector(mp_image)
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

    vf = visual_features('Data/test_video.mov',
                         'Pretrained_models/face_landmarker_v2_with_blendshapes.task',
                         'Pretrained_models/gesture_recognizer.task')
    
    resAll = vf.collect_outputs()
    resAll.to_csv('Output/test_outputs.csv', index=False)