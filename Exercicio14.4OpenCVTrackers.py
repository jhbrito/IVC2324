import cv2
# pip install opencv-contrib-python
import os
import time
import numpy as np

folder = "Files"
file = "vtest.avi"

cap = cv2.VideoCapture(os.path.join(folder, file))

tracker_types = ["KCF", "CSRT"]
tracker_type = tracker_types[0]

if tracker_type == "KCF":
    tracker = cv2.TrackerKCF_create()
if tracker_type == "CSRT":
    tracker = cv2.TrackerCSRT_create()

x, y, w, h = 495, 156, 40, 80
bbox = (x, y, w, h)

_, frame = cap.read()

img = cv2.rectangle(img=frame, pt1=(x, y), pt2=(x+w, y+h), color=255, thickness=2)

cv2.imshow("Image", img)
cv2.waitKey()
tracker.init(frame, bbox)

last_frame_timestamp = 0

while True:
    begin_time_stamp = time.time()
    framerate = 1 / (begin_time_stamp - last_frame_timestamp)
    last_frame_timestamp = begin_time_stamp

    ret, frame = cap.read()

    if ret:

        track_ok, bbox = tracker.update(frame)
        if track_ok:
            x, y, w, h = bbox
            image_show = cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=255, thickness=2)
        else:
            image_show = frame.copy()
            cv2.putText(img=image_show,
                        text="Tracking failed",
                        org=(5, 35),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(0, 0, 255),
                        thickness=2)

        text_to_show = str(int(np.round(framerate))) + " fps"
        cv2.putText(img=image_show,
                    text=text_to_show,
                    org=(5, 15),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 255, 0),
                    thickness=1)
        cv2.imshow(winname="Image", mat=image_show)

        c = cv2.waitKey(delay=100)
        if c == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
