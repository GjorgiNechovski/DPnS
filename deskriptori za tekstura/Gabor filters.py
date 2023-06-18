import cv2
import numpy as np

image1 = cv2.imread("flower1.png", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("flower2.png", cv2.IMREAD_GRAYSCALE)

image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))

tetha = 0.5  # helps find the edges
upsilon = 0.56  # helps find the angle of the edges
f_list = [0, np.pi, np.pi / 2, np.pi / 4, 3 * np.pi / 4]  # finds the center of the edge
phi = 0  # Gaussian function variable
eta_list = [2 * np.pi / 1, 2 * np.pi / 2, 2 * np.pi / 3, 2 * np.pi / 4, 2 * np.pi / 5]  # basic sharpness of the image

local_energy_list = []
mean_ampl_list = []

for theta in f_list:

    for eta in eta_list:
        kernel = cv2.getGaborKernel((3, 3), upsilon, theta, eta, tetha, phi, ktype=cv2.CV_32F)
        filtered_image1 = cv2.filter2D(image1, cv2.CV_32F, kernel) / 255.0  # get the new filtered images with the algorithm
        filtered_image2 = cv2.filter2D(image2, cv2.CV_32F, kernel) / 255.0

        mean_ampl1 = np.sum(abs(filtered_image1))
        mean_ampl2 = np.sum(abs(filtered_image2))
        mean_ampl_list.append(mean_ampl1 - mean_ampl2)

        local_energy1 = np.sum(filtered_image1 ** 2)
        local_energy2 = np.sum(filtered_image2 ** 2)
        local_energy_list.append(local_energy1 - local_energy2)

local_energy_norm = (local_energy_list - np.min(local_energy_list)) / (np.max(local_energy_list) - np.min(local_energy_list))
mean_ampl_norm = (mean_ampl_list - np.min(mean_ampl_list)) / (np.max(mean_ampl_list) - np.min(mean_ampl_list))

local_energy_norm = np.reshape(local_energy_norm, (-1, 1))
mean_ampl_norm = np.reshape(mean_ampl_norm, (-1, 1))

similarity_score = np.mean([local_energy_norm, mean_ampl_norm])

print("Similarity Score:", similarity_score)
