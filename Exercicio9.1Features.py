import cv2
import os


folder = "Files"
file = "Sharbat_Gula.jpg"

print(cv2.__version__)

image = cv2.imread(os.path.join(folder, file))
cv2.imshow("Image", image)

image_gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)

harris_cornerness = cv2.cornerHarris(src=image_gray, blockSize=2, ksize=3, k=0.04)
# normalize(src, dst[, alpha[, beta[, norm_type[, dtype[, mask]]]]]	) ->	dst
harris_cornerness_show = cv2.normalize(src=harris_cornerness,
                                       dst=None,
                                       alpha=0.0,
                                       beta=1.0,
                                       norm_type=cv2.NORM_MINMAX)
cv2.imshow("harris_cornerness_show", harris_cornerness_show)

image_harris = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
image_harris[harris_cornerness>harris_cornerness.max()*0.01] = [255, 0,  0]
cv2.imshow("image_harris", image_harris)

sift = cv2.SIFT_create()
sift_kp, sift_desc = sift.detectAndCompute(image=image, mask=None)
image_sift = image.copy()
image_sift = cv2.drawKeypoints(image=image_gray, keypoints=sift_kp, outImage=None)
cv2.imshow("image_sift", image_sift)

# surf = cv2.xfeatures2d.SURF_create(400)
# surf_kp, surf_desc = surf.detectAndCompute(image=image, mask=None)
# image_surf = image.copy()
# image_surf = cv2.drawKeypoints(image=image_gray, keypoints=surf_kp, outImage=None)
# cv2.imshow("image_surf", image_surf)

orb = cv2.ORB_create()
orb_kp, orb_desc = orb.detectAndCompute(image=image, mask=None)
image_orb = image.copy()
image_orb = cv2.drawKeypoints(image=image_gray, keypoints=orb_kp, outImage=None)
cv2.imshow("image_orb", image_orb)

brisk = cv2.BRISK_create()
brisk_kp, brisk_desc = brisk.detectAndCompute(image=image, mask=None)
image_brisk = image.copy()
image_brisk = cv2.drawKeypoints(image=image_gray, keypoints=brisk_kp, outImage=None)
cv2.imshow("image_brisk", image_brisk)

kaze = cv2.KAZE_create()
kaze_kp, kaze_desc = kaze.detectAndCompute(image=image, mask=None)
image_kaze = image.copy()
image_kaze = cv2.drawKeypoints(image=image_gray, keypoints=kaze_kp, outImage=None)
cv2.imshow("image_kaze", image_kaze)

cv2.waitKey(0)
