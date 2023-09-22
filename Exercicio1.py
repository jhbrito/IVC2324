import numpy as np
import cv2
# Declare, utilizando os tipos de dados Numpy da linguagem
# Python, um array capaz de armazenar uma imagem em
# escala de cinzentos, 8bpp, com resolução 704x576.

w = 704
h = 576

a = np.zeros((h, w), dtype="uint8")
cv2.imshow("a", a)
b = np.empty((h, w), dtype="uint8")
cv2.imshow("b", b)

c = np.zeros((h, w, 3), dtype="uint8")
# c[0:int(h/2), 0:int(w/2), 2] = 255
c[0:int(h/2), 0:int(w/2)] = [0, 0, 127]
c[0:int(h/2), int(w/2):] = [0, 127, 0]
c[int(h/2):, 0:int(w/2)] = [127, 0, 0]
c[int(h/2):, int(w/2):] = [0, 127, 127]
cv2.imshow("c", c)

d = c/255.0
cv2.imshow("d", d)

cv2.waitKey(0)
