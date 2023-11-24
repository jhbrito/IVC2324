import os.path
import cv2
import time
import numpy as np

useCamera = False

folder = "Files"
file = "vtest.avi"
# file = "slow_traffic_small.mp4"

colors = np.random.randint(0, 255, (100, 3))

#colors = (255, 255, 255)

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

# feature_params = dict(maxCorners=100,
#                      qualityLevel=0.3,
#                      minDistance=7,
#                      blockSize=7)
#p0 = cv2.goodFeaturesToTrack(prev_image, mask=None, **feature_params)
p0 = cv2.goodFeaturesToTrack(image=prev_image,
                             mask=None,
                             maxCorners=100,
                             qualityLevel=0.3,
                             minDistance=7,
                             blockSize=7)

last_frame_timestamp = 0

mask = np.zeros(image.shape, dtype=np.uint8)

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
    # image_gray = cv2.GaussianBlur(image_gray, (3, 3), 1)

    p1, st, err = cv2.calcOpticalFlowPyrLK(prevImg=prev_image,
                                             nextImg=image_gray,
                                             prevPts=p0,
                                             nextPts=None,
                                             winSize=(15, 15),
                                             maxLevel=2,
                                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        new_x, new_y = new
        old_x, old_y = old

        mask = cv2.line(img=mask,
                        pt1=(int(new_x), int(new_y)),
                        pt2=(int(old_x), int(old_y)),
                        color=colors[i].tolist(),
                        thickness=2)

    image_flow = image.copy()
    image_flow = image_flow/2 + mask/2
    image_flow = image_flow.astype(np.uint8)
    cv2.imshow("image_flow", image_flow)

    image_show = image.copy()
    text_to_show = str(int(np.round(framerate))) + " fps"
    cv2.putText(img=image_show,
                text=text_to_show,
                org=(5, 15),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 255, 0),
                thickness=1)
    cv2.imshow(winname="Image", mat=image_show)

    prev_image = image_gray
    # p0 = good_new
    p0 = good_new.reshape(-1, 1, 2)

    c = cv2.waitKey(delay=1)
    if c == 27:
        break
