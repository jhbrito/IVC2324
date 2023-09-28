import numpy as np
import cv2

im = np.ndarray((600, 800), dtype=np.uint8)
cv2.imshow("im", im)

im2 = np.zeros((600, 800), dtype=np.uint8)
cv2.imshow("im2", im2)

im3 = 255 * np.ones((600, 800), dtype=np.uint8)
cv2.imshow("im3", im3)

im4 = np.zeros((600, 800), dtype=np.float64)
cv2.imshow("im4", im4)

im5 = np.ones((600, 800), dtype=np.float64)
cv2.imshow("im5", im5)

im6 = 2.1 * np.ones((600, 800), dtype=np.float64)
cv2.imshow("im6", im6)

cv2.waitKey(0)
