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


def ivc_gray_pdf(src):
    h = np.zeros((256, ), dtype=np.float64)
    N = src.shape[0] * src.shape[1]
    for i in range(256):
        ni_m = (src==i)
        ni = np.sum(ni_m)
        h[i] = ni / N
    return h


pdf = ivc_gray_pdf(image_gray)

def ivc_pdf_2_cdf(pdf):
    cdf = np.zeros((256,), dtype=np.float64)
    cdf[0] = pdf[0]
    for i in range(1, 256):
        cdf[i] = pdf[i] + cdf[i-1]
    return cdf

cdf = ivc_pdf_2_cdf(pdf)

f = image_gray
g = np.zeros(f.shape, dtype=f.dtype)
L = 256
cdfmin = cdf[0]

for y in range(image.shape[0]):
    for x in range(image.shape[1]):
        g[y, x] = ((cdf[f[y, x]] - cdfmin) / (1 - cdfmin)) * (L - 1)
image_gray_equalized = g

pdf_equalized = ivc_gray_pdf(image_gray_equalized)
cdf_equalized = ivc_pdf_2_cdf(pdf_equalized)


plt.subplot(2, 3, 1)
plt.imshow(image_gray, cmap='gray')
plt.title("Image")
plt.axis("off")

plt.subplot(2, 3, 2)
# plt.plot(pdf)
plt.bar(range(256), pdf)
plt.title("PDF")

plt.subplot(2, 3, 3)
plt.plot(cdf)
plt.title("CDF")

plt.subplot(2, 3, 4)
plt.imshow(g, cmap="gray")
plt.title("Image Equalized")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.bar(range(256), pdf_equalized)
plt.title("PDF Equalized")

plt.subplot(2, 3, 6)
plt.plot(cdf_equalized)
plt.title("CDF Equalized")

plt.show()
cv2.waitKey(0)
