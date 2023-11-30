import os.path
import cv2
import time
import numpy as np

useCamera = False

folder = "Files"
# file = "vtest.avi"
file = "slow_traffic_small.mp4"

if useCamera:
    cap = cv2.VideoCapture()
else:
    cap = cv2.VideoCapture(os.path.join(folder, file))

cv2.namedWindow("Image")

if not cap.isOpened():
    cap.open(0)
_, image = cap.read()
if useCamera:
    image = image[:, ::-1, :]
prev_image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2HSV)

x, y, w, h = 300, 200, 100, 50
img2 = cv2.rectangle(image, (x, y), (x+w, y+h), 255, 2)
cv2.imshow("Image", img2)
cv2.waitKey()

track_window = (x, y, w, h)
roi = image[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(src=hsv_roi, lowerb=(0, 60, 32), upperb=(180, 255, 255))
roi_hist = cv2.calcHist(images=[hsv_roi], channels=[0], mask=mask, histSize=[180], ranges=(0, 180))
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

last_frame_timestamp = 0

term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    begin_time_stamp = time.time()
    framerate = 1 / (begin_time_stamp - last_frame_timestamp)
    last_frame_timestamp = begin_time_stamp

    if not cap.isOpened():
        cap.open(0)
    _, image = cap.read()
    if useCamera:
        image = image[:, ::-1, :]

    hsv = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject(images=[hsv], channels=[0], hist=roi_hist, ranges=(0, 180), scale=1)

    _, track_window = cv2.CamShift(probImage=dst, window=track_window, criteria=term_criteria)

    x, y, w, h = track_window
    image_show = cv2.rectangle(image, (x, y), (x + w, y + h), 255, 2)

    text_to_show = str(int(np.round(framerate))) + " fps"
    cv2.putText(img=image_show,
                text=text_to_show,
                org=(5, 15),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 255, 0),
                thickness=1)
    cv2.imshow(winname="Image", mat=image_show)

    c = cv2.waitKey(delay=1)
    if c == 27:
        break
