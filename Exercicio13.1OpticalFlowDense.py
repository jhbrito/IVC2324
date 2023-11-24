import os.path
import cv2
import time
import numpy as np

useCamera = True

folder = "Files"
file = "vtest.avi"
# file = "slow_traffic_small.mp4"
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
prev_image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)

last_frame_timestamp = 0
while True:
    begin_time_stamp = time.time()
    framerate = 1 / (begin_time_stamp - last_frame_timestamp)
    last_frame_timestamp = begin_time_stamp

    if not cap.isOpened():
        cap.open(0)
    _, image = cap.read()
    if useCamera:
        image = image[:, ::-1, :]

    image_gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    image_gray = cv2.GaussianBlur(image_gray, (3, 3), 1)

    flow = cv2.calcOpticalFlowFarneback(prev=prev_image,
                                        next=image_gray,
                                        flow=None,
                                        pyr_scale=0.25,
                                        levels=1,
                                        winsize=5,
                                        iterations=1,
                                        poly_n=5,
                                        poly_sigma=1.2,
                                        flags=0)
    H_float = (np.arctan(flow[:, :, 1] / flow[:, :, 0]) + np.pi) / (2 * np.pi)
    H_int = (H_float * 180).astype(np.uint8)

    flow_norm = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    S_float = cv2.normalize(src=flow_norm,
                                   dst=None,
                                   alpha=0,
                                   beta=1.0,
                                   norm_type=cv2.NORM_MINMAX)
    S_int = (S_float * 255).astype(np.uint8)

    V_int = image_gray

    image_flowHSV = image.copy()
    image_flowHSV[:, :, 0] = H_int
    image_flowHSV[:, :, 1] = S_int
    image_flowHSV[:, :, 2] = V_int

    image_flow_show = cv2.cvtColor(src=image_flowHSV, code=cv2.COLOR_HSV2BGR)

    text_to_show = str(int(np.round(framerate))) + " fps"
    cv2.putText(img=image_flow_show,
                text=text_to_show,
                org=(5, 15),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 255, 0),
                thickness=1)
    cv2.imshow(winname="Image", mat=image_flow_show)
    prev_image = image_gray

    c = cv2.waitKey(delay=1)
    if c == 27:
        break
