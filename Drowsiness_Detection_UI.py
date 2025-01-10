import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from PIL import Image
import time
import os

# Load the trained model
model = tf.keras.models.load_model('Model1WE.h5')  # Update this with the path to your model

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Define constants
DROWSINESS_THRESHOLD = 0.3
DROWSINESS_DURATION = 3  # seconds

# Define the EAR calculation function
def calculate_ear(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Define the preprocessing function
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))  # Adjust size as per your model's requirement
    image = image / 255.0  # Normalize
    return image

# Streamlit UI
st.title("Drowsiness Detection System")

# Create a camera input
camera = st.camera_input("Capture Image")

# For tracking drowsiness state
drowsy_start_time = None

if camera:
    # Convert the captured image to an OpenCV format
    img = Image.open(camera)
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Define the ROI (Region of Interest)
    x1, y1, x2, y2 = 150, 150, 550, 550
    frame = frame[y1:y2, x1:x2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Face Mesh
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            # Extract eye landmarks
            left_eye = np.array([[face_landmarks.landmark[i].x, face_landmarks.landmark[i].y]
                                 for i in [33, 160, 158, 133, 153, 144]])
            right_eye = np.array([[face_landmarks.landmark[i].x, face_landmarks.landmark[i].y]
                                  for i in [362, 385, 387, 263, 373, 380]])

            # Convert to pixel coordinates
            left_eye = (left_eye * [w, h]).astype(int)
            right_eye = (right_eye * [w, h]).astype(int)

            # Calculate EAR
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # Preprocess the frame for the model
            preprocessed_image = preprocess_image(frame)
            preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # Add batch dimension

            # Predict drowsiness
            prediction = model.predict([preprocessed_image, np.array([[ear]])])[0][0]

            # Display EAR and prediction
            st.write(f"EAR: {ear:.2f}")
            st.write(f"Prediction: {prediction:.2f}")

            if prediction < DROWSINESS_THRESHOLD:
                if drowsy_start_time is None:
                    drowsy_start_time = time.time()
                elif time.time() - drowsy_start_time >= DROWSINESS_DURATION:
                    st.warning("Drowsiness detected! Please take a break.")
                    # Optionally, play an alert sound here (e.g., using an external player)
                    drowsy_start_time = None  # Reset to ensure continuous monitoring
            else:
                drowsy_start_time = None  # Reset if not drowsy

            # Draw eye landmarks on the frame
            for (x, y) in left_eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            for (x, y) in right_eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # Convert frame back to RGB for Streamlit display
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame, channels="RGB")
else:
    st.write("No camera input detected.")
