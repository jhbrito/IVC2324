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

canny_thres_min = 150
canny_thres_max = 200

edges = cv2.Canny(image_gray,
                  threshold1=canny_thres_min,
                  threshold2=canny_thres_max)


cv2.imshow("Edges", edges)
def on_trackbar_canny_thres_min(val):
    global canny_thres_min
    canny_thres_min = val
    edges = cv2.Canny(image_gray,
                      threshold1=canny_thres_min,
                      threshold2=canny_thres_max)
    cv2.imshow("Edges", edges)


def on_trackbar_canny_thres_max(val):
    global canny_thres_max
    canny_thres_max = val
    edges = cv2.Canny(image_gray,
                      threshold1=canny_thres_min,
                      threshold2=canny_thres_max)
    cv2.imshow("Edges", edges)


cv2.createTrackbar("canny_thres_min", "Edges", 150, 255, on_trackbar_canny_thres_min)
cv2.createTrackbar("canny_thres_max", "Edges", 200, 255, on_trackbar_canny_thres_max)

cv2.waitKey(0)
