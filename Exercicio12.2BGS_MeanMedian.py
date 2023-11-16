import os.path
import cv2
import time
import numpy as np


last_frame_timestamp = 0

folder = "Files"
file = "vtest.avi"

# cap = cv2.VideoCapture(os.path.join(folder, file))
cap = cv2.VideoCapture()
if not cap.isOpened():
    cap.open(0)
_, image = cap.read()
image_gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)

threshold = 10
time_horizon = 25

previous_images = np.zeros(shape=(image_gray.shape + (time_horizon,)), dtype=image_gray.dtype)
for i in range(time_horizon):
    _, image = cap.read()
    image_gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    previous_images[:, :, i] = image_gray


def on_trackbar_threshold(val):
    global threshold
    threshold = val


cv2.namedWindow("Image")
cv2.createTrackbar("Theshold", "Image", threshold, 255, on_trackbar_threshold)

while True:
    begin_time_stamp = time.time()
    framerate = 1 / (begin_time_stamp - last_frame_timestamp)
    last_frame_timestamp = begin_time_stamp

    if not cap.isOpened():
        cap.open()
    _, image = cap.read()
    image = image[:, ::-1, :]
    image_gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)

    bg_model = np.mean(previous_images, axis=2)
    ## bg_model = np.median(previous_images, axis=2)

    dif = np.abs(image_gray - bg_model)

    _, bg_mask = cv2.threshold(src=dif, thresh=threshold, maxval=1, type=cv2.THRESH_BINARY)
    background_subtraction = image.copy()
    background_subtraction[:, :, 0] = background_subtraction[:, :, 0] * bg_mask
    background_subtraction[:, :, 1] = background_subtraction[:, :, 1] * bg_mask
    background_subtraction[:, :, 2] = background_subtraction[:, :, 2] * bg_mask
    image_out = background_subtraction.copy()

    text_to_show = str(int(np.round(framerate))) + " fps"
    cv2.putText(img=image_out,
                text=text_to_show,
                org=(5, 15),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 255, 0),
                thickness=1)
    cv2.imshow(winname="Image", mat=image_out)

    c = cv2.waitKey(delay=1)
    if c == 27:
        break
    previous_images[:, :, 0:time_horizon-1] = previous_images[:, :, 1:time_horizon]
    previous_images[:, :, -1] = image_gray
