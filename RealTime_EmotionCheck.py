
import os
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from keras.preprocessing.image import load_img, img_to_array
from keras.models import  load_model
import matplotlib.pyplot as plt
import numpy as np

# Load the pre-trained model
model = load_model("best_model.h5")

# Load the Haar cascade for face detection
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
predicted_emotion = "none"

# Open the video capture
cap = cv2.VideoCapture(0)

counter = 0

while True:
    # Read a frame from the video capture
    ret, test_img = cap.read()
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # Press 'Esc' key to exit the program
        break

    # Detect faces in the grayscale image
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
    
    for (x, y, w, h) in faces_detected:
        # Draw a rectangle around the detected face
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + h, x:x + w]  # Crop the region of interest (face area) from the image
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels /= 255  # Normalize the pixel values between 0 and 1
        stacked_img = np.expand_dims(img_pixels, axis=0)  # Shape: (1, 48, 48, 1)

        if counter == 0:
            stacked_img1 = stacked_img
            counter = counter + 1
            
        elif counter == 1:
            stacked_img2 = stacked_img
            counter = counter + 1
            
        elif counter == 2:
            stacked_img3 = stacked_img
            stacked_img = np.concatenate((stacked_img1, stacked_img2, stacked_img3), axis=0)

            stacked_img = np.reshape(stacked_img, (1, 3, 48, 48, 1))

            # Make predictions using the loaded model
            predictions = model.predict(stacked_img)

            # Find the emotion with the highest probability
            max_index = np.argmax(predictions[0])
            label_emotion_mapper = {0: "surprise", 1: "happy", 2: "anger", 3: "sadness", 4: "fear"}
            emotions = ('surprise', 'happy', 'anger', 'sadness', 'fear')
            predicted_emotion = emotions[max_index]
            print("PRED::::",predicted_emotion)
            counter = 0

        # Display the predicted emotion on the frame
        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Resize the image for display
    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis', resized_img)

    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows