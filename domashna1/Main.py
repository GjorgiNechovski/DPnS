from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('image1.jpg')
#image = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE) dobivam razlichen izgled ako go napravam ova (iako slikata e vekje crno bela)... why?

fiveMask = np.array(248, dtype=np.uint8) # 255-1-2-4 = 248 (gi kratime posledite 3 bita)
fourMask = np.array(240, dtype=np.uint8) # 255-1-2-4-8 = 240 (gi kratime posledite 4 bita itn)
threeMask = np.array(224, dtype=np.uint8)
twoMask = np.array(192, dtype=np.uint8)
oneMask = np.array(128, dtype=np.uint8)

fiveBits = cv2.bitwise_and(image, fiveMask)
# matematichko AND kade se kratat posledni 3 bita za ni ostanat prvite 5 najzvazhni bita
fourBits = cv2.bitwise_and(image, fourMask)
# matematichko AND kade se kratat posledni 4 bita za ni ostanat prvite 4 najzvazhni bita itn
threeBits = cv2.bitwise_and(image, threeMask)
twoBits = cv2.bitwise_and(image, twoMask)
oneBits = cv2.bitwise_and(image, oneMask)

plt.subplot(231), plt.imshow(fiveBits), plt.axis('off')
plt.subplot(232), plt.imshow(fourBits), plt.axis('off')
plt.subplot(233), plt.imshow(threeBits), plt.axis('off')
plt.subplot(234), plt.imshow(twoBits), plt.axis('off')
plt.subplot(235), plt.imshow(oneBits), plt.axis('off')

plt.show()