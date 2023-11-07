from ultralytics import YOLO
import os.path
import cv2
import time
import numpy as np


last_frame_timestamp = 0

folder = "Files"
file = "vtest.avi"

# cap = cv2.VideoCapture(os.path.join(folder, file))
cap = cv2.VideoCapture()

model = YOLO("yolov8n.pt")

cv2.namedWindow("Image")
while True:
    begin_time_stamp = time.time()
    framerate = 1 / (begin_time_stamp - last_frame_timestamp)
    last_frame_timestamp = begin_time_stamp

    if not cap.isOpened():
        cap.open(0)
    _, image = cap.read()
    # image = image[:, ::-1, :]
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)

    results = model(image)

    image_objects = image.copy()

    results = results[0]
    for result in results:
        b = result.boxes.data[0]
        cv2.rectangle(img=image_objects,
                      pt1=(int(b[0]), int(b[1])),
                      pt2=(int(b[2]), int(b[3])),
                      color=(255, 0, 0),
                      thickness=2)
        text = "{}:{:.2f}".format(results.names[int(b[5])], b[4])
        cv2.putText(img=image_objects,
                    text=text,
                    org=np.array(np.round((float(b[0]), float(b[1] - 1))), dtype=int),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 0, 0),
                    thickness=1)

    text_to_show = str(int(np.round(framerate))) + " fps"
    cv2.putText(img=image_objects,
                text=text_to_show,
                org=(5, 15),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 255, 0),
                thickness=1)
    image_objects = cv2.cvtColor(src=image_objects, code=cv2.COLOR_RGB2BGR)
    cv2.imshow(winname="Image", mat=image_objects)

    c = cv2.waitKey(delay=1)
    if c == 27:
        break
