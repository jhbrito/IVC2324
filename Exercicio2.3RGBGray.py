import cv2

def ivc_rgb_to_gray(src):
    r = src.copy()
    R = src[:, :, 2]
    G = src[:, :, 1]
    B = src[:, :, 0]
    gray = R * 0.299 + G * 0.587 + B * 0.114
    r[:, :, 0] = gray
    r[:, :, 1] = gray
    r[:, :, 2] = gray
    return r


cap = cv2.VideoCapture()
while True:
    if not cap.isOpened():
        cap.open(0)
    ret, image = cap.read()
    cv2.imshow("Image", image)

    image_gray = ivc_rgb_to_gray(image)
    cv2.imshow("image_gray", image_gray)
    c = cv2.waitKey(1)
    if c == 27:
        break
