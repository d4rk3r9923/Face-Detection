import cv2
import numpy as np
import pandas as pd

# Load the training data from CSV file
training_data = pd.read_csv('training_data.csv', delimiter=';', header=None)
face_images = []
face_labels = []
for i in range(len(training_data)):
    image_path = training_data.iloc[i, 0]
    label = training_data.iloc[i, 1]
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    face_images.append(image)
    face_labels.append(label)

# Train the LBPH Face Recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(face_images, np.array(face_labels))

# Initialize the webcam and Haar cascade classifier
cap = cv2.VideoCapture(0)   # "rtsp://[username]:[password]@[ip]:22345/live"
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=9)

    # Recognize faces and draw rectangles around them
    for (x, y, w, h) in faces:
        roi = gray[y:y + h, x:x + w]
        roi_resized = cv2.resize(roi, (92, 112))
        label, confidence = face_recognizer.predict(roi_resized)
        # print(confidence)
        if confidence < 90:
            if label == 0:
                name = f"Huy (ID: {label})"
            elif 1 <= label <= 40:
                name = f"s{label} (ID: {label})"
            elif label == 41:
                name = f"Hoang (ID: {label})"
            else:
                name = "Unknown"
        else:
            name = "Unknown"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('LBPH Face Recognition', frame)

    # Exit on 'esc' key press
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
