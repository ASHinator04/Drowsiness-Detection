# Drowsiness Detection Project for DRDO

## Overview

This repository contains the code and models developed during my internship at DRDO for a drowsiness detection system. The project focuses on detecting drowsiness in users by analyzing their eye aspect ratio (EAR) and facial images. The models provided here are trained to predict the drowsiness state based on these inputs.

**Important Note:** This repository only includes the code and model files. The dataset used for training the models is not included due to specific reasons, including privacy and confidentiality concerns.

---

## Repository Contents

1. **Models:**
   - The trained models are provided in `.h5` and `.keras` formats. These models are: `Model1WE.h5` and `Model1WE.keras`.
   - These models were developed during the project and can be used for inference or further fine-tuning.

2. **Data Collection Script:**
   - A Python script (`data_collection.py`) is included, which was used to collect and preprocess the data for training the models. This script captures facial images and calculates the EAR using OpenCV.

3. **Inference Script:**
   - A Python script (`inference.py`) is provided to demonstrate how to use the trained models for drowsiness detection. It takes an image of the user (resized to 224x224) and the calculated EAR as input and outputs the drowsiness state.

---

## How to Use the Models

### Input Requirements:
- **Image:** A facial image of the user resized to **224x224 pixels**.
- **EAR (Eye Aspect Ratio):** The EAR value calculated using OpenCV. The EAR is a scalar value that represents the ratio of distances between key facial landmarks around the eyes.

### Output:
- The model will output the drowsiness state of the user, which can be one of the following:
  - `0`: Not drowsy
  - `1`: Drowsy

### Data Collection Script
- The data_collection.py script is provided for reference. It demonstrates how to capture facial images and calculate EAR using OpenCV. This script was used to collect the training data for the models.

### Key Features:
- Captures facial images using a webcam.

- Calculates EAR in real-time.

- Saves images and corresponding EAR values for training.

### Disclaimer
- The dataset used to train the models is not included in this repository due to privacy and confidentiality reasons.

- The models provided are for demonstration and research purposes only. They may require further fine-tuning for specific use cases.

### Acknowledgments
- I would like to thank DRDO for providing me with the opportunity to work on this project during my internship. Special thanks to my mentors and colleagues for their guidance and support.

### License
- This project is licensed under the MIT License. See the LICENSE file for details.
