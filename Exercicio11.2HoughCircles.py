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
    global canny_thres_min, edges
    canny_thres_min = val
    edges = cv2.Canny(image_gray,
                      threshold1=canny_thres_min,
                      threshold2=canny_thres_max)
    cv2.imshow("Edges", edges)


def on_trackbar_canny_thres_max(val):
    global canny_thres_max, edges
    canny_thres_max = val
    edges = cv2.Canny(image_gray,
                      threshold1=canny_thres_min,
                      threshold2=canny_thres_max)
    cv2.imshow("Edges", edges)


cv2.createTrackbar("canny_thres_min", "Edges", 150, 255, on_trackbar_canny_thres_min)
cv2.createTrackbar("canny_thres_max", "Edges", 200, 255, on_trackbar_canny_thres_max)

cv2.waitKey(0)

dp = 1
minDist = 20
param1 = 40
param2 = 40

def update_circles():
    circles = cv2.HoughCircles(edges, method=cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist, param1=param1, param2=param2)
    circles = circles[0]
    image_circles = image.copy()
    for circle in circles:
        cv2.circle(img=image_circles,
                   center=(int(circle[0]), int(circle[1])),
                   radius=int(circle[2]),
                   color=(255, 0, 0),
                   thickness=2)
    cv2.imshow("Circulos", image_circles)


def on_trackbar_dp(val):
    global dp
    dp = val
    update_circles()


def on_trackbar_minDist(val):
    global minDist
    minDist = val
    update_circles()


def on_trackbar_param1(val):
    global param1
    param1 = val
    update_circles()


def on_trackbar_param2(val):
    global param2
    param2 = val
    update_circles()


cv2.namedWindow("Circulos")
cv2.createTrackbar("dp", "Circulos", dp, 10, on_trackbar_dp)
cv2.createTrackbar("minDist", "Circulos", minDist, 20, on_trackbar_minDist)
cv2.createTrackbar("param1", "Circulos", param1, 100, on_trackbar_param1)
cv2.createTrackbar("param2", "Circulos", param2, 100, on_trackbar_param2)

cv2.waitKey()
