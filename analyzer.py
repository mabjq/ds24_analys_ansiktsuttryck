import tensorflow as tf
import keras
import cv2
import numpy as np
import pandas as pd
import skimage as ski
import av
import tempfile
import os
from streamlit.runtime.uploaded_file_manager import UploadedFile

keras.config.disable_interactive_logging()

emotion_labels = list(map(str.lower, ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']))

class Analyzer:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.results = pd.DataFrame(columns=["frame"] + emotion_labels + ["x", "y", "width", "height"])
        self.last_faces = []  # Spara senaste ansikten och etiketter

    def analyze(self, 
                model=None,
                file: UploadedFile | None = None, 
                skip: int | None = 1,
                confidence: float | None = .5) -> tuple[bool, pd.DataFrame]:
        if file is None:
            print("No file provided.")
            return False, self.results
        
        # Spara uppladdad fil temporärt
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(file.read())
            temp_file_path = tmp_file.name

        print(f"Temporary file saved at: {temp_file_path}")
        
        cap = cv2.VideoCapture(temp_file_path)
        if not cap.isOpened():
            print("Failed to open video file.")
            os.remove(temp_file_path)
            return False, self.results

        # Sätt upp videoutdata
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        print(f"Video dimensions: {frame_width}x{frame_height}, FPS: {fps}")
        
        out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        if not out.isOpened():
            print("Failed to initialize VideoWriter.")
            cap.release()
            os.remove(temp_file_path)
            return False, self.results

        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if i % skip == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

                self.last_faces = []  # Återställ för varje analys
                for (x, y, width, height) in faces:
                    roi = gray[y:y+height, x:x+width]
                    roi = cv2.resize(roi, (48, 48))
                    roi = roi / 255.0
                    roi = np.expand_dims(roi, axis=(0, -1))
                    
                    prediction = model.predict(roi, verbose=0)[0]
                    print(f"Raw prediction for frame {i}: {prediction}")
                    
                    prediction_binary = (prediction > confidence).astype(int)
                    
                    frame_results = pd.Series(
                        np.concatenate([[i], prediction_binary, [x, y, width, height]]), 
                        index=self.results.columns)
                    self.results = pd.concat([self.results, frame_results.to_frame().T])

                    # Skapa etikett för visning
                    emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
                    max_idx = np.argmax(prediction)
                    max_emotion = emotions[max_idx]
                    max_prob = prediction[max_idx]
                    label = f"{max_emotion}: {max_prob:.2f}"
                    self.last_faces.append((x, y, width, height, label))

            # Rita alltid de senaste ansiktena
            for (x, y, width, height, label) in self.last_faces:
                cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            out.write(frame)
            i += 1

        cap.release()
        out.release()
        os.remove(temp_file_path)
        print("Processing complete. Output video saved as 'output_video.mp4'.")
        return True, self.results