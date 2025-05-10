import tensorflow as tf
import keras
import cv2
import numpy as np
import pandas as pd
import skimage as ski
import av
from streamlit.runtime.uploaded_file_manager import UploadedFile

keras.config.disable_interactive_logging()

emotion_labels = list(map(str.lower, ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']))

class Analyzer:
    """ Audience Emotion Analyzer class """
    def __init__(self) -> None:
        self.detector = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
        self.results = pd.DataFrame(columns=["frame"] + emotion_labels + ["x", "y", "width", "height"])

    def analyze(self, 
        model = None,
        file: UploadedFile | None = None, 
        skip: int | None = 1,
        confidence: float | None = .5) -> tuple[bool, pd.DataFrame]:
        
        """ Analyze video in file """
        # Check for file
        if file is None:
            raise ValueError('Must have a file to analyze.')

        # Load video and get properties
        container = av.open(file, mode="r")
        stream = container.streams.video[0]

        i = 0

        for i, frame in enumerate(container.decode(stream)):
            # Convert to grayscale for cv
            gray = frame.to_ndarray()
            frame = frame.to_rgb().to_ndarray()

            # Detect faces
            faces = self.detector.detectMultiScale(gray)

            # Classify emotions in detected faces
            for face in faces:
                x, y, width, height = face

                # Preprocess detected face pt. I 
                roi_gray = gray[y:y+height, x:x+width]
                roi_gray = ski.transform.resize(roi_gray, (48, 48))

                if (i % skip == 0) & (np.sum([roi_gray]) != 0):

                    # Preprocess detected face pt. II
                    roi = roi_gray.astype('float') / 255.0
                    roi = keras.utils.img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)
                    roi = tf.convert_to_tensor(roi)

                    # Predict emotion
                    prediction = model.predict(roi, verbose = 0)[0] > confidence

                    # Check for confident prediction, add to results
                    if sum(prediction > 0):
                        frame_results = pd.Series(
                            np.concatenate([[i],  
                                            prediction, 
                                            [x, y, width, height]]), 
                                            index=self.results.columns)
                        self.results = pd.concat([self.results, frame_results.to_frame().T])

            i += 1

        container.close()

        return True, self.results