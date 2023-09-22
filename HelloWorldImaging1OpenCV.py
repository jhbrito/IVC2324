# Demo with a few examples of using OpenCV functions and UI
# uses lena: https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png

import numpy as np
import cv2
import os.path
import urllib.request as urllib_request

print("OpenCV Version:", cv2.__version__)

image = np.ones((256, 256), dtype="uint8")
image = image * 127
image[0:128, 0:128] = 0
image[128:, 128:] = 255
cv2.imshow("Image", image)
cv2.waitKey(0)

# Opening and Viewing an Image
folder = "Files"
if not os.path.isfile(os.path.join(folder, "lena.png")):
    urllib_request.urlretrieve("https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png",
                               os.path.join(folder, "lena.png"))

image = cv2.imread(os.path.join(folder, "lena.png"))
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imshow("Image RGB", rgb_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


viewImage(image, "Lena")

# Edit pixels
edited = image.copy()
edited[200:390, 200:360, 0] = 255
viewImage(edited, "Lena edited")

# Cropping
cropped = image[200:390, 200:360]
viewImage(cropped, "Lena cropped")

# Resizing
scale_percent = 10  # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
viewImage(resized, "Lena resized to {}%".format(scale_percent))

# Drawing a Rectangle
output = image.copy()
# cv2.rectangle(output, (200, 200), (360, 390), (255, 0, 0), 10)
cv2.rectangle(img=output, pt1=(200, 200), pt2=(360, 390), color=(255, 0, 0), thickness=10)
viewImage(output, "Lena with a rectangle")

# Drawing a line
cv2.line(output, (256, 390), (256, 512), (0, 0, 255), 5)
viewImage(output, "Lena with a line")

# Writing on an image
cv2.putText(output, "Lena", (360, 390), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
viewImage(output, "Lena with text")

# Saving an image
cv2.imwrite("./output.jpg", output)

# Rotating
(h, w, d) = image.shape
center = (w // 2, h // 2)
rot = 45
M = cv2.getRotationMatrix2D(center, rot, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
viewImage(rotated, "Lena rotated by {} degrees".format(rot))

# Blend
alpha_slider_max = 100
alpha_slider_init = 50

def on_trackbar_weight(val):
    alpha = val / alpha_slider_max
    beta = (1.0 - alpha)
    blend = cv2.addWeighted(image, alpha, rotated, beta, 0.0)
    cv2.imshow('Lena blended', blend)


window_name = 'Lena blended'
cv2.namedWindow(window_name)
trackbar_name = 'Alpha 0 - {}'.format(alpha_slider_max)
cv2.createTrackbar(trackbar_name, window_name, alpha_slider_init, alpha_slider_max, on_trackbar_weight)
on_trackbar_weight(alpha_slider_init)
cv2.waitKey()
cv2.destroyWindow('Lena blended')
