# Age-Gender-Detection-CNN

Real-Time Age and Gender Detection System

A Deep Learning Computer Vision Project

## 📌 Overview

Estimating a person's age and gender from facial images has wide applications in security, marketing, and human-computer interaction. This project presents a real-time age and gender detection system using a Convolutional Neural Network (CNN) trained on the UTKFace dataset.

The system captures a live webcam feed, detects faces, and predicts the age and gender of each detected person in real time.

## 🎯 Objectives

- Detect faces from a live webcam feed in real-time
- Predict age and gender using a trained CNN model
- Build a practical, end-to-end deep learning application
- Provide a reusable, retrainable model pipeline

## 🧠 Technologies Used

- Python
- TensorFlow / Keras (CNN model training and inference)
- OpenCV (face detection and webcam frame processing)
- NumPy (data handling and preprocessing)
- Jupyter Notebook (training, retraining, and prediction workflows)

## ⚙️ How It Works

1. Load and preprocess the UTKFace dataset (face images labeled with age and gender)
2. Train a CNN model to learn facial features correlated with age and gender
3. Save the trained model (age_gender_model.h5)
4. Capture a live webcam feed using OpenCV
5. Detect faces in each frame
6. Pass each detected face through the trained model
7. Display the predicted age and gender on the live video feed

## 📊 Model Pipeline

Webcam Frame → Face Detection → Preprocessing → CNN Model → Age & Gender Prediction

The CNN takes a cropped, resized face image as input and outputs two predictions:
- Age: estimated as a numeric value
- Gender: classified as male/female

## 📁 Project Structure

Age-Gender-Detection-CNN/
├── train_model.ipynb       (Trains the CNN on the UTKFace dataset)
├── retrain.ipynb           (Retrains/fine-tunes the model)
├── webcam_predict.ipynb    (Runs real-time prediction using webcam)
├── age_gender_model.h5     (Saved trained model)
└── README.md

## 📂 Dataset

This project uses the UTKFace dataset (https://susanqq.github.io/UTKFace/), which contains over 20,000 face images labeled with age, gender, and ethnicity.

To retrain the model yourself, download the dataset separately and place it in a folder named UTKFace/ in the project root before running train_model.ipynb.

## 🚀 How to Run

1. Clone the repository

git clone https://github.com/Kashish-Bhatt/Age-Gender-Detection-CNN.git

2. Install dependencies

pip install tensorflow opencv-python numpy

3. To run prediction using the pre-trained model

jupyter notebook webcam_predict.ipynb

4. To retrain the model from scratch (optional)

jupyter notebook train_model.ipynb

## 🔮 Future Improvements

- Improve age prediction accuracy with a larger, more diverse dataset
- Add emotion detection alongside age and gender
- Deploy as a web app using Flask or Streamlit
