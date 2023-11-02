import cv2 as cv2
import numpy as np

imagem1 = cv2.imread("Files/Boat1.jpg")
imagem2 = cv2.imread("Files/Boat2.jpg")

s1 = imagem1.shape
s2 = imagem2.shape

imagem_dual = np.zeros((s1[0], s1[1]+s2[1], s1[2]), dtype=np.uint8)
imagem_dual[:,0:s1[1],:] = imagem1
imagem_dual[:,s2[1]:,:] = imagem2

cv2.namedWindow("Imagem", 0)
cv2.imshow("Imagem", imagem_dual)
cv2.waitKey(0)

sift = cv2.SIFT_create()
sift_kp1, sift_desc1 = sift.detectAndCompute(imagem1, None)
sift_kp2, sift_desc2 = sift.detectAndCompute(imagem2, None)

imagem1_sift = imagem1.copy()
imagem1_sift = cv2.drawKeypoints(imagem1_sift, sift_kp1, None)
cv2.namedWindow("imagem1_sift", 0)
cv2.imshow("imagem1_sift", imagem1_sift)

imagem2_sift = imagem2.copy()
imagem2_sift = cv2.drawKeypoints(imagem2_sift, sift_kp2, None)
cv2.namedWindow("imagem2_sift", 0)
cv2.imshow("imagem2_sift", imagem2_sift)

cv2.waitKey(0)

distances = np.zeros((len(sift_kp1), len(sift_kp2)), dtype=np.float64)

print(len(sift_kp1))

for i in range(len(sift_kp1)):
    if not (i % 100):
        print(i)
    for j in range(len(sift_kp2)):
        desc_i_p1 = sift_desc1[i, :]
        desc_j_p2 = sift_desc2[j, :]
        dif = desc_i_p1 - desc_j_p2
        abs_dif = np.abs(dif)
        sum_abs_dif = np.sum(abs_dif)
        distances[i, j] = sum_abs_dif

cv2.waitKey(0)
