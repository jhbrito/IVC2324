import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

folder = "Files"
file = "moedas.jpg"

image = cv2.imread(os.path.join(folder, file))
cv2.imshow("Image", image)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_gray = image_gray / 255.0

cv2.imshow("Image Gray", image_gray)

Mx_Prewitt = np.array([[-1, 0, 1],
               [-1, 0, 1],
               [-1, 0, 1]],
              dtype=np.float64)
My_Prewitt = np.array([[-1, -1, -1],
               [0, 0, 0],
               [1, 1, 1]],
              dtype=np.float64)

dx_Prewit = cv2.filter2D(src=image_gray,
                  ddepth=-1,
                  kernel=Mx_Prewitt)
cv2.imshow("dx Prewit", dx_Prewit)
dy_Prewit = cv2.filter2D(src=image_gray,
                  ddepth=-1,
                  kernel=My_Prewitt)
cv2.imshow("dy Prewit", dy_Prewit)

gradient_Prewit = np.sqrt(dx_Prewit ** 2 + dy_Prewit ** 2)
cv2.imshow("gradient Prewit", gradient_Prewit)

dir_Prewit = np.arctan(dy_Prewit/dx_Prewit)

def on_trackbar(val):
    threshold = val/100.0
    ret, thresholded_gradient = cv2.threshold(src=gradient_Prewit,
                                              thresh=threshold,
                                              maxval=1.0,
                                              type=cv2.THRESH_BINARY)
    cv2.imshow("Threshold Prewit", thresholded_gradient)
    pass

cv2.namedWindow("Threshold Prewit")
cv2.createTrackbar("T", "Threshold Prewit", 50, 100, on_trackbar)

Mx_Sobel = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]],
              dtype=np.float64)
My_Sobel = np.array([[-1, -2, -1],
               [0, 0, 0],
               [1, 2, 1]],
              dtype=np.float64)
dx_Sobel = cv2.filter2D(src=image_gray,
                  ddepth=-1,
                  kernel=Mx_Sobel)
cv2.imshow("dx Sobel", dx_Sobel)
dy_Sobel = cv2.filter2D(src=image_gray,
                  ddepth=-1,
                  kernel=My_Sobel)
cv2.imshow("dy Sobel", dy_Sobel)

gradient_Sobel = np.sqrt(dx_Sobel ** 2 + dy_Sobel ** 2)
cv2.imshow("gradient Sobel", gradient_Sobel)

dir_Sobel = np.arctan(dy_Sobel/dx_Sobel)

def on_trackbar(val):
    threshold = val/100.0
    ret, thresholded_gradient = cv2.threshold(src=gradient_Sobel,
                                              thresh=threshold,
                                              maxval=1.0,
                                              type=cv2.THRESH_BINARY)
    cv2.imshow("Threshold Sobel", thresholded_gradient)
    pass

cv2.namedWindow("Threshold Sobel")
cv2.createTrackbar("T", "Threshold Sobel", 50, 100, on_trackbar)


cv2.waitKey(0)
