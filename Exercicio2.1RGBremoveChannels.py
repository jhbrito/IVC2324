import numpy as np
import cv2
import os.path
import urllib.request as urllib_request

print("OpenCV Version:", cv2.__version__)

# Opening and Viewing an Image
folder = "Files"
if not os.path.isfile(os.path.join(folder, "lena.png")):
    urllib_request.urlretrieve("https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png",
                               os.path.join(folder, "lena.png"))

image = cv2.imread(os.path.join(folder, "lena.png"))
cv2.imshow("Image", image)

print(image.shape)
def ivc_rgb_remove_red(src):
    r = src.copy()
    # r[:, :, 2] = np.zeros((src.shape[0], src.shape[1]), dtype=np.uint8)
    r[:, :, 2] = 0
    return r

def ivc_rgb_remove_green(src):
    r = src.copy()
    r[:, :, 1] = 0
    return r

def ivc_rgb_remove_blue(src):
    r = src.copy()
    r[:, :, 0] = 0
    return r


image_sem_red = ivc_rgb_remove_red(image)
image_sem_green = ivc_rgb_remove_green(image)
image_sem_blue = ivc_rgb_remove_blue(image)
cv2.imshow("image_sem_red", image_sem_red)
cv2.imshow("image_sem_green", image_sem_green)
cv2.imshow("image_sem_blue", image_sem_blue)
cv2.waitKey(0)
