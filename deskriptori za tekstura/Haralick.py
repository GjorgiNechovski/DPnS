import cv2
import numpy as np

import mahotas.features.texture as mht


def extract_features(image):
    textures = mht.haralick(image, ignore_zeros=True)
    ht_mean = textures.mean(axis=0)  # average for each characteristic
    return ht_mean


def compare_images(image_path1, image_path2):
    image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

    image1 = cv2.resize(image1, (800, 600))
    image2 = cv2.resize(image2, (800, 600))

    features1 = extract_features(image1)

    features2 = extract_features(image2)

    similarity_score = np.linalg.norm(features1 - features2)  # Euclidean distance

    normalized_score = 1 / (1 + similarity_score)

    return normalized_score


image_path1 = "discord1.png"
image_path2 = "discord2.png"

score = compare_images(image_path1, image_path2)

print("Similarity Score: {:.5f}".format(score))
