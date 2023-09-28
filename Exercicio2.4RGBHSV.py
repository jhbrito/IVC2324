import cv2

def ivc_rgb_to_hsv(src):
    r = src.copy()
    R = src[:, :, 2]
    G = src[:, :, 1]
    B = src[:, :, 0]
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

    return hsv


cap = cv2.VideoCapture()
while True:
    if not cap.isOpened():
        cap.open(0)
    ret, image = cap.read()
    cv2.imshow("Image", image)

    image_hsv = ivc_rgb_to_hsv(image)
    image_hsv[:, :, 0] = 60
    image_bgr_outravez = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow("image_bgr_outravez", image_bgr_outravez)
    c = cv2.waitKey(1)
    if c == 27:
        break
