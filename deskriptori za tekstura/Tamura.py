import cv2
import numpy as np


def coarseness(image, kmax):
    image = np.array(image)
    w = image.shape[0]
    h = image.shape[1]
    kmax = kmax if (np.power(2, kmax) < w) else int(np.log(w) / np.log(2))  # check if kmax is valid
    kmax = kmax if (np.power(2, kmax) < h) else int(np.log(h) / np.log(2))  # if not make it take it's value by using log(2) on it
    average_gray = np.zeros([kmax, w, h])  # create 4 empty matrix
    horizontal = np.zeros([kmax, w, h])
    vertical = np.zeros([kmax, w, h])
    Sbest = np.zeros([w, h])

    for k in range(kmax):  # iterations over kmax
        window = np.power(2, k)
        for wi in range(w)[window:(w - window)]:
            for hi in range(h)[window:(h - window)]:  # calculate averages over the height in gray color
                average_gray[k][wi][hi] = np.sum(image[wi - window:wi + window, hi - window:hi + window])
        for wi in range(w)[window:(w - window - 1)]:
            for hi in range(h)[window:(h - window - 1)]:  # calculate averages over the width in gray color
                horizontal[k][wi][hi] = average_gray[k][wi + window][hi] - average_gray[k][wi - window][hi]
                vertical[k][wi][hi] = average_gray[k][wi][hi + window] - average_gray[k][wi][hi - window]
        horizontal[k] = horizontal[k] * (1.0 / np.power(2, 2 * (k + 1)))
        vertical[k] = vertical[k] * (1.0 / np.power(2, 2 * (k + 1)))

    for wi in range(w):  # find the best values from the gray colors in height and width
        for hi in range(h):
            h_max = np.max(horizontal[:, wi, hi])
            h_max_index = np.argmax(horizontal[:, wi, hi])
            v_max = np.max(vertical[:, wi, hi])
            v_max_index = np.argmax(vertical[:, wi, hi])
            index = h_max_index if (h_max > v_max) else v_max_index
            Sbest[wi][hi] = np.power(2, index)

    fcrs = np.mean(Sbest)
    return fcrs


def contrast(image):
    image = np.array(image)
    image = np.reshape(image, (1, image.shape[0] * image.shape[1]))
    m4 = np.mean(np.power(image - np.mean(image), 4)) # calculate 4th moment fr gray color
    v = np.var(image)
    std = np.power(v, 0.5)  # deviation
    alfa4 = m4 / np.power(v, 2)
    fcon = std / np.power(alfa4, 0.25)  # 4th central moment of gray color
    return fcon


def directionality(image):
    image = np.array(image, dtype='int64')
    h = image.shape[0]
    w = image.shape[1]
    convH = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    convV = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    deltaH = np.zeros([h, w])
    deltaV = np.zeros([h, w])
    theta = np.zeros([h, w])

    # calc for deltaH
    for hi in range(h)[1:h - 1]:  # use the horizontal Sobel's filter
        for wi in range(w)[1:w - 1]:
            img = image[hi - 1:hi + 2, wi - 1:wi + 2]
            deltaH[hi][wi] = np.sum(convH * img)

    # calc for deltaV
    for hi in range(h)[1:h - 1]:  # use the vertical Sobel's filter
        for wi in range(w)[1:w - 1]:
            img = image[hi - 1:hi + 2, wi - 1:wi + 2]
            deltaV[hi][wi] = np.sum(convV * img)

    # calc for theta
    for hi in range(h):  # create a histogram for, the Sobel's filters
        for wi in range(w):
            if deltaH[hi][wi] != 0:
                theta[hi][wi] = np.arctan(deltaV[hi][wi] / deltaH[hi][wi])
            else:
                theta[hi][wi] = np.pi / 2

    histogram = np.zeros(4)
    for hi in range(h):
        for wi in range(w):
            if -np.pi / 8 < theta[hi][wi] <= np.pi / 8:
                histogram[0] += 1
            elif np.pi / 8 < theta[hi][wi] <= 3 * np.pi / 8:
                histogram[1] += 1
            elif -3 * np.pi / 8 < theta[hi][wi] <= -np.pi / 8:
                histogram[2] += 1
            else:
                histogram[3] += 1

    histogram /= np.sum(histogram)
    fdir = -np.sum(histogram * np.log2(histogram))  # point of direction
    return fdir


def roughness(fcrs, fcon):
    f_r = np.sqrt(np.power(fcrs, 2) + np.power(fcon, 2))
    return f_r


def calculate_similarity(img1, img2):
    fcrs1 = coarseness(img1, 5)
    fcon1 = contrast(img1)
    fdir1 = directionality(img1)
    f_r1 = roughness(fcrs1, fcon1)

    fcrs2 = coarseness(img2, 5)
    fcon2 = contrast(img2)
    fdir2 = directionality(img2)
    f_r2 = roughness(fcrs2, fcon2)

    # Calculate Euclidean distance between the feature vectors
    distance = np.sqrt((fcrs1 - fcrs2) ** 2 + (fcon1 - fcon2) ** 2 + (fdir1 - fdir2) ** 2 + (f_r1 - f_r2) ** 2)

    # Normalize the distance to obtain a similarity score between 0 and 1
    similarity = 1 / (1 + distance)

    return similarity


if __name__ == '__main__':
    image1_path = "flower1.png"
    image2_path = "flower2.png"

    img1 = cv2.imread(image1_path)
    img1 = cv2.resize(img1, (800, 600))
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    img2 = cv2.imread(image2_path)
    img2 = cv2.resize(img2, (800, 600))
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    similarity_score = calculate_similarity(img1, img2)

    print("Similarity Score: {:.4f}".format(similarity_score))
