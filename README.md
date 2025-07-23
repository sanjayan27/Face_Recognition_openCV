# Face_Recognition_openCV
# 🧠 OpenCV Projects – Face Recognition & Motion Detection

Welcome to my OpenCV-based computer vision projects! This repository includes two main applications:

1. **Face Detection & Recognition using Haar Cascade + FisherFace Algorithm**
2. **Moving Object Detection using Contour Analysis**

---

## 📌 Table of Contents

- [About](#about)
- [Tech Stack](#tech-stack)
- [Features](#features)
- [Project 1: Face Recognition](#face-recognition)
- [Project 2: Moving Object Detection](#moving-object-detection)
- [How to Run](#how-to-run)
- [Results](#results)
- [Future Enhancements](#future-enhancements)
- [Author](#author)

---

## 📖 About

This project demonstrates how computer vision can be used to solve real-world problems using Python and OpenCV.  
It includes:

- Real-time **face detection and recognition** using pre-trained Haar Cascade and the FisherFace algorithm.
- Real-time **motion detection** using background subtraction and contours.

---

## 🛠 Tech Stack

- 🐍 Python
- 🖼 OpenCV (`cv2`)
- 🧠 Haar Cascade Classifier
- 🧪 FisherFace Recognizer
- 💻 Webcam Integration (Live Feed)

---

## ✨ Features

### ✅ Face Recognition
- Detects faces using Haar Cascades.
- Recognizes faces using the FisherFace algorithm.
- Supports real-time webcam input.

### ✅ Moving Object Detection
- Detects motion using contour-based analysis.
- Highlights moving objects in live video feed.
- Filters noise using thresholding and Gaussian blur.

---

## 🧑‍💻 Face Recognition

- Uses `haarcascade_frontalface_default.xml` for face detection.
- Trains a FisherFace recognizer on labeled face images.
- Capable of recognizing known individuals with accuracy.

### Key Functions:
```python
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
