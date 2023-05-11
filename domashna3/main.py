import os

import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

inputFolder = "database"
outputFolder = "outputFolder"

images = os.listdir(inputFolder)

for i in images:
    path = os.path.join(inputFolder, i)
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (3, 3), 0)

    _, threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    morphed_image = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel=np.ones((5, 5), np.uint8))

    contour, _ = cv2.findContours(morphed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    finalImage = cv2.drawContours(np.zeros(morphed_image.shape, np.uint8), contour, -1, 255, 1)

    finalImage = cv2.dilate(finalImage, (1, 1))

    outputPath = os.path.join(outputFolder, i)
    cv2.imwrite(outputPath, finalImage)

#     cv2.imshow("something", finalImage)
#     cv2.waitKey(0)
#
# cv2.destroyAllWindows()

