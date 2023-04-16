import cv2
import numpy as np
import pandas as pd
import dlib

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

# Train the Eigen Face Recognizer
face_recognizer = cv2.face.EigenFaceRecognizer_create()
face_recognizer.train(face_images, np.array(face_labels))

# Initialize the webcam and ERT facial landmark detector
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image using ERT
    faces = detector(gray)

    # Recognize faces and draw rectangles and landmarks around them
    for face in faces:
        landmarks = predictor(gray, face)
        face_img = np.array(gray)[face.top():face.bottom(), face.left():face.right()]
        face_img = cv2.resize(face_img, (92, 112))
        label, confidence = face_recognizer.predict(face_img)
        if confidence < 4500:
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
        for i in range(68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)
        cv2.putText(frame, name, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Facial Landmark Detection with ERT', frame)

    # Exit on 'esc' key press
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
