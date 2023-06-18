import cv2
import numpy as np


def compute_glcm(image, dx, dy):  # dx and dy direction we want to go to search for neighbouring pixels
    rows, cols = image.shape
    glcm = np.zeros((256, 256))

    for i in range(rows - dx):
        for j in range(cols - dy):
            i_index = image[i, j]
            j_index = image[i + dx, j + dy]
            glcm[i_index, j_index] += 1

    glcm /= glcm.sum()

    return glcm


def extract_features(image):
    glcm = compute_glcm(image, 1, 0)
    xs = np.sum(glcm, axis=1)  # Dissimilarity
    ys = np.sum(glcm * np.arange(256)[:, np.newaxis], axis=1)  # Correlation
    bs = np.sum(glcm * (np.arange(256)[:, np.newaxis] - np.arange(256)) ** 2, axis=1)  # Contrast
    cs = np.sum(glcm ** 2, axis=1)  # Energy
    ds = np.sum(glcm / (1 + (np.arange(256)[:, np.newaxis] - np.arange(256)) ** 2), axis=1)  # Homogeneity

    return xs, ys, bs, cs, ds


image_path1 = "flower1.png"
image_path2 = "flower2.png"

image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

xs1, ys1, bs1, cs1, ds1 = extract_features(image1)

xs2, ys2, bs2, cs2, ds2 = extract_features(image2)

dissimilarity = np.sqrt(
    np.sum((np.concatenate((xs1, ys1, bs1, cs1, ds1)) - np.concatenate((xs2, ys2, bs2, cs2, ds2))) ** 2))

similarity = 1 / (1 + dissimilarity)

print("Similarity score: ", similarity)
