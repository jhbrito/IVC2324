import cv2
import os

import numpy as np

folder = "Files"
file = "moedas.jpg"

image = cv2.imread(os.path.join(folder, file))
cv2.imshow("Image", image)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Image Gray", image_gray)

thres = 100


def on_trackbar_thres(val):
    global thres
    thres = val
    _, image_thresholded = cv2.threshold(
        image_gray, thres, 255, type=cv2.THRESH_BINARY)
    cv2.imshow("Threshold", image_thresholded)

    contours, hierarchy = cv2.findContours(image=image_thresholded,
                                           mode=cv2.RETR_TREE,
                                           method=cv2.CHAIN_APPROX_NONE)
    image_contours = np.zeros(image_gray.shape, np.uint8)
    # cv2.drawContours(image=image_contours, contours=contours,
    #                  contourIdx=-1, color=255, thickness=-1)
    cv2.drawContours(image=image_contours, contours=contours,
                    contourIdx=-1, color=128, thickness=-1)
    cv2.imshow("Contours", image_contours)

    image_contours_color = np.zeros(image.shape, np.uint8)
    # cv2.drawContours(image=image_contours_color, contours=contours,
    #                  contourIdx=-1, color=(0, 255, 255), thickness=-1)
    # cv2.drawContours(image=image_contours_color, contours=contours,
    #                  contourIdx=3, color=(0, 255, 255), thickness=-1)
    cv2.drawContours(image=image_contours_color, contours=contours,
                     contourIdx=-1, color=(0, 255, 0), thickness=-1,
                     hierarchy=hierarchy, maxLevel=2)
    cv2.imshow("Contours Color", image_contours_color)

    image_contour_0 = np.zeros(image.shape, np.uint8)
    cv2.drawContours(image=image_contour_0, contours=contours,
                     contourIdx=0, color=(0, 0, 255), thickness=-1)
    cv2.imshow("Contour 0", image_contour_0)
    contour_0 = contours[0]
    M = cv2.moments(contour_0)
    Cx = int(np.round(M['m10']/M['m00']))
    Cy = int(np.round(M['m01']/M['m00']))
    print("Cx:", Cx, " ; Cy:", Cy)
    contour_0_area = cv2.contourArea(contour_0)
    print("Contour 0 Area:", contour_0_area)
    perimeter = cv2.arcLength(curve=contour_0, closed=True)
    print("Perimeter", perimeter)


cv2.namedWindow("Threshold")
cv2.createTrackbar("threshold", "Threshold",
                   thres, 255, on_trackbar_thres)

cv2.waitKey(0)
