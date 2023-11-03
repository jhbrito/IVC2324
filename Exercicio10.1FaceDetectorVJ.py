import cv2
import os
import time
import numpy as np


last_frame_timestamp = 0

cap = cv2.VideoCapture()

classifier_folder = "C:/Users/jhasb/.conda/envs/IVC2324/Lib/site-packages/cv2/data"
classifier_file = "haarcascade_frontalface_default.xml"
classifier_file_path = os.path.join(classifier_folder, classifier_file)
face_detector = cv2.CascadeClassifier(classifier_file_path)

cv2.namedWindow("Image")
while True:
    begin_time_stamp = time.time()
    framerate = 1 / (begin_time_stamp - last_frame_timestamp)
    last_frame_timestamp = begin_time_stamp

    if not cap.isOpened():
        cap.open(0)
    _, image = cap.read()
    image = image[:, ::-1, :]

    faces = face_detector.detectMultiScale(image=image, scaleFactor=1.1)
    image_faces = image.copy()
    for face in faces:
        (x, y, w, h) = face
        cv2.rectangle(img=image_faces, pt1=(x, y),
                      pt2=(x+w, y+h),
                      color=(255, 0, 0), thickness=2)

    text_to_show = str(int(np.round(framerate))) + " fps"
    cv2.putText(img=image_faces, text=text_to_show,
                org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 255, 0), thickness=2)

    cv2.imshow("Image", image_faces)

    c = cv2.waitKey(1)
    if c == 27:
        break

