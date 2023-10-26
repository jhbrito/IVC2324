import cv2
import os

import numpy as np

folder = "Files"
file = "moedas.jpg"

image = cv2.imread(os.path.join(folder, file))
cv2.imshow("Image", image)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Image Gray", image_gray)

thres = 127


def on_trackbar_thres(val):
    global thres
    thres = val
    _, image_thresholded = cv2.threshold(
        image_gray, thres, 255, type=cv2.THRESH_BINARY)
    cv2.imshow("Threshold", image_thresholded)

    kernel_dilate = np.ones((3, 3), dtype=np.uint8)
    mask_dilate = cv2.dilate(src=image_thresholded, kernel=kernel_dilate)
    cv2.imshow("Mask Dilate", mask_dilate)

    kernel_erode = np.ones((3, 3), dtype=np.uint8)
    mask_erode = cv2.erode(src=image_thresholded, kernel=kernel_erode)
    cv2.imshow("Mask Erode", mask_erode)

    kernel_open = kernel_erode
    mask_open = cv2.morphologyEx(src=image_thresholded,
                                 op=cv2.MORPH_OPEN, kernel=kernel_open)
    cv2.imshow("Mask Open", mask_open)

    kernel_close = kernel_open
    mask_close = cv2.morphologyEx(src=image_thresholded,
                                 op=cv2.MORPH_CLOSE, kernel=kernel_close)
    cv2.imshow("Mask Close", mask_close)

cv2.namedWindow("Threshold")
cv2.createTrackbar("threshold", "Threshold",
                   thres, 255, on_trackbar_thres)

cv2.waitKey(0)
