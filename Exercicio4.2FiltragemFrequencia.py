import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

folder = "Files"
file = "lena.png"
# file = "baboon.png"

image = cv2.imread(os.path.join(folder, file))
cv2.imshow("Image", image)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_gray = image_gray / 255.0

cv2.imshow("Image Gray", image_gray)

image_gray_fft = np.fft.fft2(image_gray)
image_gray_fft_v = np.abs(image_gray_fft) / np.mean(np.abs(image_gray_fft))
cv2.imshow("Image Gray FFT", image_gray_fft_v)

image_gray_fft_shift = np.fft.fftshift(image_gray_fft)
image_gray_fft_shift_v = np.abs(image_gray_fft_shift) / np.mean(np.abs(image_gray_fft_shift))
cv2.imshow("Image Gray FFT Shift", image_gray_fft_shift_v)

tamanho = image_gray_fft_shift.shape
filtro_low_pass = np.zeros(tamanho, dtype=np.float64)
filtro_high_pass = np.zeros(tamanho, dtype=np.float64)

raio_maximo = tamanho[1]/2.0
raio = 0.25 * raio_maximo
centro_x = tamanho[1]/2.0
centro_y = tamanho[0]/2.0

for y in range(tamanho[0]):
    for x in range(tamanho[1]):
        d = np.sqrt((x-centro_x) ** 2 + (y-centro_y) ** 2)
        if d < raio:
            filtro_low_pass[y, x] = 1.0
        else:
            filtro_high_pass[y, x] = 1.0

cv2.imshow("Low Pass Filter", filtro_low_pass)
cv2.imshow("High Pass Filter", filtro_high_pass)

image_gray_fft_shift_filtered = image_gray_fft_shift * \
                                filtro_low_pass
image_gray_fft_shift_filtered_v = np.abs(image_gray_fft_shift_filtered) / np.mean(np.abs(image_gray_fft_shift_filtered))
cv2.imshow("Image Gray FFT Shift Filtered", image_gray_fft_shift_filtered_v)

image_gray_fft_shift_filtered_high = image_gray_fft_shift * \
                                filtro_high_pass
image_gray_fft_shift_filtered_high_v = np.abs(image_gray_fft_shift_filtered_high) / np.mean(np.abs(image_gray_fft_shift_filtered_high))
cv2.imshow("Image Gray FFT Shift Filtered High", image_gray_fft_shift_filtered_high_v)

image_gray_fft_shift_filtered_unshift = \
    np.fft.ifftshift(image_gray_fft_shift_filtered)
image_gray_fft_shift_filtered_unshift_v = \
    np.abs(image_gray_fft_shift_filtered_unshift) / \
    np.mean(np.abs(image_gray_fft_shift_filtered_unshift))
cv2.imshow("Image Gray FFT Shift Filtered Unshift", image_gray_fft_shift_filtered_unshift_v)

image_gray_fft_shift_filtered_high_unshift = \
    np.fft.ifftshift(image_gray_fft_shift_filtered_high)
image_gray_fft_shift_filtered_high_unshift_v = \
    np.abs(image_gray_fft_shift_filtered_high_unshift) / \
    np.mean(np.abs(image_gray_fft_shift_filtered_high_unshift))
cv2.imshow("Image Gray FFT Shift Filtered High Unshift", image_gray_fft_shift_filtered_high_unshift_v)

image_gray_fft_shift_filtered_unshift_ifft = \
    np.fft.ifft2(image_gray_fft_shift_filtered_unshift)
image_gray_fft_shift_filtered_unshift_ifft = \
    np.abs(image_gray_fft_shift_filtered_unshift_ifft)
cv2.imshow("Image Gray FFT Shift Filtered Unshift IFFT", image_gray_fft_shift_filtered_unshift_ifft)

image_gray_fft_shift_filtered_high_unshift_ifft = \
    np.fft.ifft2(image_gray_fft_shift_filtered_high_unshift)
image_gray_fft_shift_filtered_high_unshift_ifft = \
    np.abs(image_gray_fft_shift_filtered_high_unshift_ifft)
cv2.imshow("Image Gray FFT Shift Filtered High Unshift IFFT", image_gray_fft_shift_filtered_high_unshift_ifft)

cv2.waitKey(0)
