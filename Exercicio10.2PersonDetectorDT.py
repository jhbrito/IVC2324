import os.path

import cv2
import time
import numpy as np


last_frame_timestamp = 0

folder = "Files"
file = "vtest.avi"

cap = cv2.VideoCapture(os.path.join(folder, file))

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.namedWindow("Image")
while True:
    begin_time_stamp = time.time()
    framerate = 1 / (begin_time_stamp - last_frame_timestamp)
    last_frame_timestamp = begin_time_stamp

    if not cap.isOpened():
        cap.open(0)
    _, image = cap.read()
    # image = image[:, ::-1, :]

    persons, _ = hog.detectMultiScale(img=image, winStride=(8, 8), scale=1.1)

    image_persons = image.copy()
    for person in persons:
        (x, y, w, h) = person
        cv2.rectangle(img=image_persons, pt1=(x, y),
                      pt2=(x+w, y+h),
                      color=(255, 0, 0), thickness=2)

    text_to_show = str(int(np.round(framerate))) + " fps"
    cv2.putText(img=image_persons, text=text_to_show,
                org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 255, 0), thickness=2)
    cv2.imshow("Image", image_persons)

    c = cv2.waitKey(1)
    if c == 27:
        break
