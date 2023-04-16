import cv2
import os
import csv

folder_name = 'training_data'
num_people = int(input("Variant(s) = "))
num_images = int(input("Image(s) per Variant = "))

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)
target = []
name = []

for i in range(num_people):

    target_name = input(f"Enter a label for person {i + 1}: ")
    target_id = input(f"Enter ID for person {i + 1}: ")
    name.append(target_name)
    target.append(target_id)

    print(f"Capturing training data for person {target_name}. Press 'space' to take a picture.")

    subfolder_name = f'{folder_name}/{target_name}'
    os.makedirs(subfolder_name, exist_ok=True)

    image_id = 1
    while image_id <= num_images:

        ret, frame = cap.read()

        cv2.imshow('Training Data', frame)

        if cv2.waitKey(1) & 0xFF == ord(' '):
            image_filename = f'{subfolder_name}/{target_name}_{image_id}.jpg'
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 1:
                (x, y, w, h) = faces[0]
                cropped = gray[y:y+h, x:x+w]
                resized_cropped = cv2.resize(cropped, (92, 112), interpolation=cv2.INTER_AREA)
                cv2.imwrite(image_filename, resized_cropped)
                print(f'Saved image {image_id} for person {target_name}.')
                image_id += 1
            else:
                print("No face or multiple faces detected in the image. Please try again.")

        if cv2.waitKey(1) & 0xFF == 27:
            break

with open('training_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        for subdir, dirs, files in os.walk(folder_name):
            for i in files:
                if i.endswith('.jpg') or i.endswith('.jpeg') or i.endswith('.png'):
                    for variant in range(len(target)):
                        label = os.path.basename(subdir).replace(name[variant], str(target[variant]))
                    label = os.path.basename(subdir).replace('s', '').replace('Huy', '0').replace('Hoang', '41')
                    image_path = os.path.join(subdir, i).replace(os.sep, '/')
                    writer.writerow([image_path, label])
                    # print(f"Added {image_path};{label} to CSV file.")
print(f"Image paths have been extracted to {folder_name}.csv.")
cap.release()
cv2.destroyAllWindows()
