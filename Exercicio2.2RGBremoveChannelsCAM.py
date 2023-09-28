import cv2

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


cap = cv2.VideoCapture()
while True:
    if not cap.isOpened():
        cap.open(0)
    ret, image = cap.read()
    cv2.imshow("Image", image)

    image_sem_red = ivc_rgb_remove_red(image)
    image_sem_green = ivc_rgb_remove_green(image)
    image_sem_blue = ivc_rgb_remove_blue(image)
    cv2.imshow("image_sem_red", image_sem_red)
    cv2.imshow("image_sem_green", image_sem_green)
    cv2.imshow("image_sem_blue", image_sem_blue)
    c = cv2.waitKey(1)
    if c == 27:
        break
