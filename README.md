# weather-Classification
Weather classification model (sunny weather vs rainy weather) trained using Google Teachable Machine and implemented in Python with Keras.
Weather Classification

This project classifies weather into two categories: “Sunny” or “Rainy” based on image data. It uses a model trained with Google Teachable Machine and implemented in Python using Keras and TensorFlow.

# Project Overview

The goal of this project is to classify images of weather conditions as either sunny or rainy. The model is trained using Google Teachable Machine, a tool that allows for easy image classification model creation. Once trained, the model is exported and used within a Python script, leveraging Keras and TensorFlow for prediction.

# Requirements

To run this project, you will need to have the following installed:
 • Python 3.x
 • Keras
 • TensorFlow
 • OpenCV
 • NumPy

You can install the necessary dependencies using the following:
pip install -r requirements.txt

# Installation and Usage
 1. Clone the Repository:
Clone this repository to your local machine using the following command:
https://github.com/Raghad1110/weather-Classification.git
2. Install Dependencies:
Install all required Python libraries listed in the requirements.txt file:
pip install -r requirements.txt
3. Run the Model:
To classify an image, run the following script:
python classify_weather.py
The script will read an input image, preprocess it, and predict whether the weather is sunny or rainy.

5. Model Output:
The output of the model will display the classification result on the screen, showing whether the weather is sunny or rainy based on the input image.

 # Example Usage
Here is a Python code snippet for loading the model and using it to classify a weather image:
import cv2
from keras.models import load_model
import numpy as np

# Load the pre-trained model
model = load_model('keras_model.h5')

# Load labels (Sunny, Rainy)
with open('labels.txt', 'r') as file:
    labels = file.readlines()

# Load the input image
image = cv2.imread('test_image.jpg')
image = cv2.resize(image, (224, 224))  # Resize to match input size of the model
image = np.expand_dims(image, axis=0)  # Add batch dimension

# Make a prediction
prediction = model.predict(image)
predicted_label = labels[np.argmax(prediction)]  # Get the label with highest probability

# Output the prediction result
print(f'Predicted Weather: {predicted_label}')
