import cv2

cap = cv2.VideoCapture()

while True:
    if not cap.isOpened():
        cap.open(0)
    ret, image = cap.read()
    cv2.imshow("Image", image)

    image_inverted = image.copy()
    image_inverted = image[:, ::-1, :]
    # for y in range(image.shape[0]):
    #     for x in range(image.shape[1]):
    #         for c in range(image.shape[2]):
    #             image_inverted[y, x, c] = image[y, image.shape[1] -x -1, c]

    cv2.imshow("Inverted", image_inverted)
    c = cv2.waitKey(1)
    if c == 27:
        break

if cap.isOpened():
    cap.release()
cv2.destroyAllWindows()
