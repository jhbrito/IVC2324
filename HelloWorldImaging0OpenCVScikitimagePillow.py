# Demo with a few examples of using OpenCV, scikit-image and Pillow
# packages: opencv-python, scikit-image, Pillow
# uses lena: https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png
import os.path
import urllib.request as urllib_request
import matplotlib.pyplot as plt
import cv2
from skimage import io as skimage_io
import PIL

# Opening and Viewing an Image
folder = "Files"
if os.path.isfile(os.path.join(folder, "lena.png")):
    print("Test Image File exist")
else:
    print("Test Image File does not exist; downloading...")
    urllib_request.urlretrieve("https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png",
                               os.path.join(folder, "lena.png"))

image_opencv = cv2.imread(os.path.join(folder, "lena.png"))
image_skimage = skimage_io.imread(os.path.join(folder, "lena.png"))
image_pil = PIL.Image.open(os.path.join(folder, "lena.png"))
image_pil.show()

plt.subplot(2, 2, 1)
plt.imshow(image_opencv)
plt.title("OpenCV {} (BGR)".format(type(image_opencv)))
plt.axis("off")

image_opencv_rgb = cv2.cvtColor(image_opencv, cv2.COLOR_BGR2RGB)
plt.subplot(2, 2, 2)
plt.imshow(image_opencv_rgb)
plt.title("OpenCV {} (RGB)".format(type(image_opencv_rgb)))
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(image_skimage)
plt.title("Scikit-image {} (RGB)".format(type(image_skimage)))
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(image_pil)
plt.title("Pillow {} (RGB)".format(type(image_pil)))
plt.axis("off")
plt.show()
