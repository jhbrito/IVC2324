import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

folder = "Files"
file = "lena.png"
image = cv2.imread(os.path.join(folder, file))
cv2.imshow("Image", image)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Image Gray", image_gray)

kernel_3x3 = (1.0/9) * np.ones((3, 3), dtype=np.float64)
image_filtered_3x3 = cv2.filter2D(src=image_gray, ddepth=-1, kernel=kernel_3x3)
cv2.imshow("Filtro Media 3x3", image_filtered_3x3)

image_filtered_3x3_b = cv2.blur(image_gray, (3, 3))
cv2.imshow("Filtro Media 3x3 b", image_filtered_3x3_b)


kernel_5x5 = (1.0/25) * np.ones((5, 5), dtype=np.float64)
image_filtered_5x5 = cv2.filter2D(src=image_gray, ddepth=-1, kernel=kernel_5x5)
cv2.imshow("Filtro Media 5x5", image_filtered_5x5)

kernel_7x7 = (1.0/49) * np.ones((7, 7), dtype=np.float64)
image_filtered_7x7 = cv2.filter2D(src=image_gray, ddepth=-1, kernel=kernel_7x7)
cv2.imshow("Filtro Media 7x7", image_filtered_7x7)

kernel_9x9 = (1.0/81) * np.ones((9, 9), dtype=np.float64)
image_filtered_9x9 = cv2.filter2D(src=image_gray, ddepth=-1, kernel=kernel_9x9)
cv2.imshow("Filtro Media 9x9", image_filtered_9x9)

image_filtered_median_3 = cv2.medianBlur(image_gray, 3)
cv2.imshow("Filtro Mediana 3x3", image_filtered_median_3)

image_filtered_median_5 = cv2.medianBlur(image_gray, 5)
cv2.imshow("Filtro Mediana 5x5", image_filtered_median_5)

image_filtered_gaussian_3 = cv2.GaussianBlur(image_gray, (3, 3), 1)
cv2.imshow("Filtro Gaussiana 3x3", image_filtered_gaussian_3)

image_filtered_gaussian_5 = cv2.GaussianBlur(image_gray, (5, 5), 1)
cv2.imshow("Filtro Gaussiana 5x5", image_filtered_gaussian_5)

kernel_passa_alto_a = (1.0/6.0) * np.array([[0, -1, 0],
                                          [-1, 4, -1],
                                          [0, -1, 0]])
image_filtered_passa_alto_a = cv2.filter2D(src=image_gray,
                                           ddepth=-1,
                                           kernel=kernel_passa_alto_a)
cv2.imshow("Filtro Passa Alto A", image_filtered_passa_alto_a)

kernel_passa_alto_b = (1.0/9.0) * np.array([[-1, -1, -1],
                                          [-1, 8, -1],
                                          [-1, -1, -1]])
image_filtered_passa_alto_b = cv2.filter2D(src=image_gray,
                                           ddepth=-1,
                                           kernel=kernel_passa_alto_b)
cv2.imshow("Filtro Passa Alto B", image_filtered_passa_alto_b)

kernel_passa_alto_c = (1.0/16.0) * np.array([[-1, -2, -1],
                                          [-2, 12, -2],
                                          [-1, -2, -1]])
image_filtered_passa_alto_c = cv2.filter2D(src=image_gray,
                                           ddepth=-1,
                                           kernel=kernel_passa_alto_c)
cv2.imshow("Filtro Passa Alto C", image_filtered_passa_alto_c)

cv2.waitKey(0)
