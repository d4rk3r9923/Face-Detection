import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

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

# Preprocess the data
face_images = np.array(face_images)
face_labels = np.array(face_labels)
face_images = face_images.reshape((-1, 92, 112, 1))
face_images = face_images.astype('float32') / 255.0

# Define the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(92, 112, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(42)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(face_images, face_labels, epochs=10, batch_size=32)

# Initialize the webcam and Haar cascade classifier
cap = cv2.VideoCapture(0)  # "rtsp://[username]:[password]@[ip]:22345/live"
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
        roi_resized = roi_resized.reshape((-1, 92, 112, 1))
        roi_resized = roi_resized.astype('float32') / 255.0
        label_logits = model.predict(roi_resized)
        label = np.argmax(label_logits)
        confidence = tf.nn.softmax(label_logits)[0][label] * 100
        if confidence < 50:
            name = "Unknown"
        else:
            if label == 0:
                name = f"Huy (ID: {label})"
            elif 1 <= label <= 40:
                name = f"s{label} (ID: {label})"
            elif label == 41:
                name = f"Hoang (ID: {label})"
            else:
                name = "Unknown"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('FaceNet (CNN) Face Recognition', frame)

    # Exit on 'esc' key press
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
