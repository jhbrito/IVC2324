import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

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


cv2.namedWindow("Threshold")
cv2.createTrackbar("threshold", "Threshold",
                   thres, 255, on_trackbar_thres)

thres_media = np.mean(image_gray)
print("Media global:", thres_media)
_, image_thresholded_media = cv2.threshold(
    image_gray, thres_media, 255, type=cv2.THRESH_BINARY)
cv2.imshow("Threshold Media Global", image_thresholded_media)

image_thresholded_adaptive_mean = \
    cv2.adaptiveThreshold(src=image_gray,
                          maxValue=255,
                          adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                          thresholdType=cv2.THRESH_BINARY,
                          blockSize=11,
                          C=-5)
cv2.imshow("Adaptive Mean", image_thresholded_adaptive_mean)

image_thresholded_adaptive_gaussian = \
    cv2.adaptiveThreshold(src=image_gray,
                          maxValue=255,
                          adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                          thresholdType=cv2.THRESH_BINARY,
                          blockSize=11,
                          C=-5)
cv2.imshow("Adaptive Gaussian", image_thresholded_adaptive_gaussian)

ret, image_thresholded_otsu = cv2.threshold(
    src=image_gray, thresh=0, maxval=1,
    type=(cv2.THRESH_BINARY | cv2.THRESH_OTSU))
cv2.imshow("Otsu", image_thresholded_otsu * 255)
print("Optimum Threshold (Otsu):", ret)

image_segmented_otsu = image_gray * image_thresholded_otsu
cv2.imshow("Segmented Otsu", image_segmented_otsu)

cv2.waitKey(0)
