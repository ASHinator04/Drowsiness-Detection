import cv2
import numpy as np
import mediapipe as mp
import os
import time
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

# Define constants
DROWSINESS_THRESHOLD = 0.3
ALERT_THRESHOLD = 0.9
DROWSINESS_DURATION = 3  # in seconds

# Load your pre-trained model
model = load_model('Model1WE.h5')

# Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)
    return img

def calculate_ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

class DrowsinessDetector:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)
        self.canvas = tk.Canvas(window, width=400, height=400)
        self.canvas.pack()
        
        self.btn_start = Button(window, text="Start Detection", width=30, command=self.start_detection)
        self.btn_start.pack(anchor=tk.CENTER, expand=True)

        self.btn_stop = Button(window, text="Stop Detection", width=30, command=self.stop_detection)
        self.btn_stop.pack(anchor=tk.CENTER, expand=True)

        self.drowsy_start_time = None
        self.detecting = False
        self.update()

        self.window.mainloop()

    def start_detection(self):
        self.detecting = True

    def stop_detection(self):
        self.detecting = False
        self.drowsy_start_time = None

    def update(self):
        ret, frame = self.vid.read()
        
        if ret:
            x1, y1 = 150, 150
            x2, y2 = 550, 550
            frame = frame[y1:y2, x1:x2]

            if self.detecting:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        left_eye = np.array([[face_landmarks.landmark[i].x, face_landmarks.landmark[i].y]
                                             for i in [33, 160, 158, 133, 153, 144]])
                        right_eye = np.array([[face_landmarks.landmark[i].x, face_landmarks.landmark[i].y]
                                              for i in [362, 385, 387, 263, 373, 380]])

                        h, w, _ = frame.shape
                        left_eye = (left_eye * [w, h]).astype(int)
                        right_eye = (right_eye * [w, h]).astype(int)

                        left_ear = calculate_ear(left_eye)
                        right_ear = calculate_ear(right_eye)

                        ear = (left_ear + right_ear) / 2.0

                        preprocessed_image = preprocess_image(frame)
                        prediction = model.predict([np.expand_dims(preprocessed_image, axis=0), np.array([[ear]])])[0][0]

                        cv2.putText(frame, f'Prediction: {prediction:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f'EAR: {ear:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        if prediction < DROWSINESS_THRESHOLD:
                            if self.drowsy_start_time is None:
                                self.drowsy_start_time = time.time()
                            elif time.time() - self.drowsy_start_time >= DROWSINESS_DURATION:
                                os.system('after-explosion-ears-ringing-95784.mp3')  # Replace with the command to play your mp3 file
                        else:
                            self.drowsy_start_time = None

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img_tk = ImageTk.PhotoImage(image=img)

            self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.window.after(10, self.update)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Create a window and pass it to the Application object
App(tk.Tk(), "Real-Time Drowsiness Detection")
