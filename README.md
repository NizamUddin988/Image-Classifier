# Image-Classifier
Developed an Image Classifier website for my Machine Learning project that predicts whether an uploaded picture is of a dog, a cat, or an unknown object. The application is built using HTML, CSS, and Python.

##Features

- Dog Detection - Accurately identifies dog breeds and images

- Cat Recognition - Detects various cat breeds and images

- Unknown Object Detection - Identifies non-dog/cat images

- Easy Upload - Simple drag-and-drop or file selection

- Fast Prediction - Quick inference with optimized model

- Responsive Design - Works on all devices

- Modern UI - Clean and user-friendly interface

- Confidence Scores - Shows prediction confidence levels

- Multiple Formats - Supports JPG, PNG, JPEG images

- Client-Side Processing - Optional browser-based processing

##Technology Stack

1. Backend - Python 3.8+, TensorFlow/Keras, Flask/FastAPI, OpenCV, NumPy, Pillow

2. Frontend - HTML5,  CSS3 , JavaScript

3.Machine Learning - CNN Architecture, Transfer Learning, Image Preprocessing, Model Optimization

🚀 Quick Start

  ##Installation & Setup

1. Clone the repository

    git clone https://github.com/NizamUddin988/Image-Classifier.git
    cd Image-Classifier

2. Dependencies

    pip install -r requirements.txt
   
    pip install tensorflow
   
    pip install flask
   
    pip install opencv-python
   
    pip install pillow
   
    pip install numpy

3.Train the model
  python train_model.py
     
    ##Expected output during training:

       Epoch 1/10
      187/187 [==============================] - 45s 240ms/step - loss: 0.6932 - accuracy: 0.5012 - val_loss: 0.6931 - val_accuracy: 0.5000
      Epoch 2/10
      187/187 [==============================] - 40s 214ms/step - loss: 0.6543 - accuracy: 0.6123 - val_loss: 0.6014 - val_accuracy: 0.6789
      ...
      Model saved as: model.h5
      Training completed!

4.Run python server

  python main.py

