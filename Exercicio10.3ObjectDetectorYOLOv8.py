from ultralytics import YOLO
import os.path
import cv2
import time
import numpy as np

useCamera = True

folder = "Files"
file = "vtest.avi"

if useCamera:
    cap = cv2.VideoCapture()
else:
    cap = cv2.VideoCapture(os.path.join(folder, file))

model = YOLO("yolov8n.pt")
print("Known classes ({})".format(len(model.names)))
for i in range(len(model.names)):
    print("{} : {}".format(i, model.names[i]))

cv2.namedWindow("Image")

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
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)

    results = model(image, verbose=False)

    image_objects = image.copy()

    objects = results[0]
    for object in objects:
        box = object.boxes.data[0]
        pt1 = (int(box[0]), int(box[1]))
        pt2 = (int(box[2]), int(box[3]))
        confidence = box[4]
        class_id = int(box[5])
        if confidence > 0.8:  # class_id == 0 and confidence > 0.8:
            cv2.rectangle(img=image_objects, pt1=pt1, pt2=pt2, color=(255, 0, 0), thickness=2)
            text = "{}:{:.2f}".format(objects.names[class_id], confidence)
            cv2.putText(img=image_objects,
                        text=text,
                        org=np.array(np.round((float(box[0]), float(box[1] - 1))), dtype=int),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(255, 0, 0),
                        thickness=1)
        else:
            cv2.rectangle(img=image_objects,
                          pt1=(int(box[0]), int(box[1])),
                          pt2=(int(box[2]), int(box[3])),
                          color=(0, 0, 0),
                          thickness=2)


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
