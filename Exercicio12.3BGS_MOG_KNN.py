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

cv2.namedWindow("Image MOG")
cv2.namedWindow("Image KNN")

backsub_mog = cv2.createBackgroundSubtractorMOG2()
N = backsub_mog.getNMixtures()
# backsub_mog.setNMixtures(5)

backsub_knn = cv2.createBackgroundSubtractorKNN()
k = backsub_knn.getkNNSamples()
# backsub_knn.setkNNSamples(25)

while True:
    begin_time_stamp = time.time()
    framerate = 1 / (begin_time_stamp - last_frame_timestamp)
    last_frame_timestamp = begin_time_stamp

    if not cap.isOpened():
        cap.open()
    _, image = cap.read()
    image = image[:, ::-1, :]

    fg_mask_mog = backsub_mog.apply(image)
    cv2.imshow(winname="fg_mask_mog", mat=fg_mask_mog)

    fg_mask_knn = backsub_knn.apply(image)
    cv2.imshow(winname="fg_mask_knn", mat=fg_mask_knn)

    background_subtraction_mog = image.copy()
    _, fg_mask_mog = cv2.threshold(src=fg_mask_mog, thresh=200, maxval=1, type=cv2.THRESH_BINARY)
    background_subtraction_mog[:, :, 0] = background_subtraction_mog[:, :, 0] * fg_mask_mog
    background_subtraction_mog[:, :, 1] = background_subtraction_mog[:, :, 1] * fg_mask_mog
    background_subtraction_mog[:, :, 2] = background_subtraction_mog[:, :, 2] * fg_mask_mog
    image_out_mog = background_subtraction_mog.copy()

    background_subtraction_knn = image.copy()
    _, fg_mask_knn = cv2.threshold(src=fg_mask_knn, thresh=200, maxval=1, type=cv2.THRESH_BINARY)
    background_subtraction_knn[:, :, 0] = background_subtraction_knn[:, :, 0] * fg_mask_knn
    background_subtraction_knn[:, :, 1] = background_subtraction_knn[:, :, 1] * fg_mask_knn
    background_subtraction_knn[:, :, 2] = background_subtraction_knn[:, :, 2] * fg_mask_knn
    image_out_knn = background_subtraction_knn.copy()

    text_to_show = str(int(np.round(framerate))) + " fps"
    cv2.putText(img=image_out_mog,
                text=text_to_show,
                org=(5, 15),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 255, 0),
                thickness=1)
    cv2.imshow(winname="Image MOG", mat=image_out_mog)
    cv2.imshow(winname="Image KNN", mat=image_out_knn)

    c = cv2.waitKey(delay=1)
    if c == 27:
        break
