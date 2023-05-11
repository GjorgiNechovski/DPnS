import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("image1.jpg")
image = cv2.GaussianBlur(image, (5, 5), 0)

compasKernels = [
    np.array([[1, 1, 0],
              [1, 0, -1],
              [0, -1, -1]]),

    np.array([[1, 1, 1],
              [0, 0, 0],
              [-1, -1, -1]]),

    np.array([[0, 1, 1],
              [-1, 0, 1],
              [-1, -1, 0]]),

    np.array([[1, 0, -1],
              [1, 0, -1],
              [1, 0, -1]]),

    np.array([[-1, 0, 1],
              [-1, 0, 1],
              [-1, 0, 1]]),

    np.array([[0, -1, -1],
              [1, 0, -1],
              [1, 1, 0]]),

    np.array([[-1, -1, -1],
              [0, 0, 0],
              [1, 1, 1]]),

    np.array([[-1, -1, 0],
             [-1, 0, 1],
             [0, 1, 1]])
]

detectedEdges = []
subplots = []

for i, kernel in enumerate(compasKernels):
    detectedEdges.append(
        cv2.filter2D(image, -1, kernel))

    subplot = plt.subplot(3, 3, i+1)
    subplot.imshow(detectedEdges[i])
    subplot.axis("off")
    subplots.append(subplot)

subplotsForThreshold = []

threshold = float(input("Vnesete threshold: "))

for i, kernel in enumerate(compasKernels):
    kernel = kernel*threshold

filtered_edges = np.array(detectedEdges) * threshold
allKernels = np.maximum.reduce(filtered_edges)

subplot = plt.subplot(3, 3, 9)
subplot.imshow(allKernels)
subplot.axis("off")
subplot.set_title("Threshold: 0.1")
subplots.append(subplot)

#plt.savefig("detectedEdges")
plt.show()

