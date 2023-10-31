import cv2
import numpy as np

hmin = 70
hmax = 100
smin = 50
smax = 255
vmin = 50
vmax = 255

cap = cv2.VideoCapture()
if not cap.isOpened():
    cap.open(0)
ret, image = cap.read()
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def update_segmentation():
    if hmin < hmax:
        ret, mask_hmin = cv2.threshold(src=image_hsv[:, :, 0], thresh=hmin-1,
                                       maxval=1, type=cv2.THRESH_BINARY)
        ret, mask_hmax = cv2.threshold(src=image_hsv[:, :, 0], thresh=hmax,
                                      maxval=1, type=cv2.THRESH_BINARY_INV)
        mask_h = mask_hmin * mask_hmax
    else:
        ret, mask_hmin = cv2.threshold(src=image_hsv[:, :, 0], thresh=hmin,
                                       maxval=1, type=cv2.THRESH_BINARY)
        ret, mask_hmax = cv2.threshold(src=image_hsv[:, :, 0], thresh=hmax-1,
                                       maxval=1, type=cv2.THRESH_BINARY_INV)
        mask_h = cv2.bitwise_or(mask_hmin, mask_hmax)


    ret, mask_smin = cv2.threshold(src=image_hsv[:, :, 1], thresh=smin-1,
                                   maxval=1, type=cv2.THRESH_BINARY)
    ret, mask_smax = cv2.threshold(src=image_hsv[:, :, 1], thresh=smax,
                                  maxval=1, type=cv2.THRESH_BINARY_INV)
    mask_s = mask_smin * mask_smax


    ret, mask_vmin = cv2.threshold(src=image_hsv[:, :, 2], thresh=vmin-1,
                                   maxval=1, type=cv2.THRESH_BINARY)
    ret, mask_vmax = cv2.threshold(src=image_hsv[:, :, 2], thresh=vmax,
                                  maxval=1, type=cv2.THRESH_BINARY_INV)
    mask_v = mask_vmin * mask_vmax

    # cv2.imshow("Mask Hmin", mask_hmin*255)
    # cv2.imshow("Mask Hmax", mask_hmax * 255)
    # cv2.imshow("Mask H", mask_h * 255)
    # cv2.imshow("Mask S", mask_s * 255)
    # cv2.imshow("Mask V", mask_v * 255)
    mask = mask_h * mask_s * mask_v
    cv2.imshow("Mask", mask * 255)

    contours, hierarchy = cv2.findContours(image=mask,
                                           mode=cv2.RETR_TREE,
                                           method=cv2.CHAIN_APPROX_NONE)

    mask_filtered = np.zeros(mask.shape, np.uint8)
    for i in range(len(contours)):
        contour = contours[i]
        contour_area = cv2.contourArea(contour)
        if contour_area > 100:
            cv2.drawContours(image=mask_filtered, contours=contours,
                             contourIdx=i, color=1, thickness=-1)
            M = cv2.moments(contour)
            Cx = int(np.round(M['m10'] / M['m00']))
            Cy = int(np.round(M['m01'] / M['m00']))
            perimeter = cv2.arcLength(curve=contour, closed=True)
            if Cx > (2/3) * mask.shape[1]:
                cv2.rectangle(img=mask_filtered,
                              pt1=(mask.shape[1]-10, 0),
                              pt2=(mask.shape[1], mask.shape[0]),
                              color=1, thickness=6)
            elif Cx < (1/3) * mask.shape[1]:
                cv2.rectangle(img=mask_filtered,
                              pt1=(0, 0),
                              pt2=(10, mask.shape[0]),
                              color=1, thickness=6)
            else:
                pass
    cv2.imshow("Mask Filtered", mask_filtered * 255)


def on_change_hmin(val):
    global hmin
    hmin = val
    update_segmentation()


def on_change_hmax(val):
    global hmax
    hmax = val
    update_segmentation()


def on_change_smin(val):
    global smin
    smin = val
    update_segmentation()


def on_change_smax(val):
    global smax
    smax = val
    update_segmentation()


def on_change_vmin(val):
    global vmin
    vmin = val
    update_segmentation()


def on_change_vmax(val):
    global vmax
    vmax = val
    update_segmentation()


cv2.namedWindow("Image")
cv2.createTrackbar("Hmin", "Image", hmin, 180, on_change_hmin)
cv2.createTrackbar("Hmax", "Image", hmax, 180, on_change_hmax)
cv2.createTrackbar("Smin", "Image", smin, 255, on_change_smin)
cv2.createTrackbar("Smax", "Image", smax, 255, on_change_smax)
cv2.createTrackbar("Vmin", "Image", vmin, 255, on_change_vmin)
cv2.createTrackbar("Vmax", "Image", vmax, 255, on_change_vmax)


while True:
    if not cap.isOpened():
        cap.open(0)
    _, image = cap.read()
    image = image[:, ::-1, :]
    cv2.imshow("Image", image)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    update_segmentation()
    c = cv2.waitKey(1)
    if c == 27:
        break
