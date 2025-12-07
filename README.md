# ASL-Sign-Language-Recognition-Using-Mediapipe-Deep-Learning
A deep learning–based ASL sign recognition system using Mediapipe hand landmarks. The model is trained on extracted keypoints instead of images, enabling fast and accurate real-time recognition through webcam. The project includes data preprocessing, model training, and live prediction.

---

## 📸 Project Overview

- Extract 21 hand landmarks using MediaPipe (each landmark has X, Y, Z → 63 features total).
- Train a Deep Learning model using the keypoints dataset.
- Save the trained model (`asl_keypoints_model.h5`) and label classes (`label_classes.npy`).
- Use a real-time Python script (`SignLanguageDetector.py`) to recognize signs from camera input.

---

## 🧰 Technologies Used

- **Python**
- **Mediapipe**
- **TensorFlow / Keras**
- **NumPy**
- **Pandas**
- **Scikit-Learn**
- **OpenCV**

---

# How to Run the Project

## 1) Create & Activate Virtual Environment
python -m venv venv

# Windows:
venv\Scripts\activate

---------------------------------------

## 2) Install Required Libraries
pip install mediapipe opencv-python numpy pandas scikit-learn tensorflow

# If TensorFlow causes issues:
pip install tensorflow==2.12

---------------------------------------

## 3) Place the Dataset
Make sure the file:
asl_mediapipe_keypoints_dataset.csv
is in the same folder as:

- train_model.py
- SignLanguageDetector.py
- venv/

---------------------------------------

## 4) Train the Model
Run:
python train_model.py

This will generate:
- asl_keypoints_model.h5
- label_classes.npy

---------------------------------------

## 5) Run Real-Time Detection
Run:
python SignLanguageDetector.py

Press Q to exit the webcam.

---------------------------------------

## Notes
- Ensure your camera is working before running detection.
- Keep all files in the same directory.
- Activate the venv before running any script.


## 🧠 Model Architecture

The neural network (MLP) consists of:

- Dense layer (256 neurons, ReLU)
- Dropout (0.3)
- Dense layer (128 neurons, ReLU)
- Dropout (0.2)
- Dense layer (64 neurons, ReLU)
- Output layer (`num_classes` neurons, Softmax)

Loss: `categorical_crossentropy`  
Optimizer: `Adam`  
Batch Size: `32`  
Epochs: `25`

---

## Team Members

153 – Ziad Mohamed Ramadan Abdelhadi  
243 – Omar Elsayed Mokhtar Suleiman  
272 – Karim Ashraf Elsayed Elmetwally Batikh  
289 – Mohamed Ibrahim Mostafa Mohamed Selim  
346 – Mohamed Yehia Zakaria Mohamed Zayed  
439 – Yehia Mohamed Youssef Amin Al-Lawati  
448 – Youssef Ragab Fouad El-Sayed Gomaa Al-Attarah  
449 – Youssef Saber Ibrahim Hassan Bahbah  

---

## Instructor
Dr. Samar El-Badawihi  

## Course
Computer Vision and Robotics
