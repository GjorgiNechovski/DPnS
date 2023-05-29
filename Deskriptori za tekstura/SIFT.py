import cv2
import numpy as np

# Load images
image1 = cv2.imread("flower1.png", cv2.IMREAD_COLOR)
image2 = cv2.imread("flower2.png", cv2.IMREAD_COLOR)

# Resize images to have the same dimensions
image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))

# Convert images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Detect keypoints and compute descriptors using SIFT
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# Create BFMatcher object
bf = cv2.BFMatcher()

# Match descriptors
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Apply ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append([m])

# Calculate similarity score
similarity = len(good_matches) / max(len(keypoints1), len(keypoints2))

# Print similarity score
print("Similarity score:", similarity)

# Draw keypoints and lines on images
image_matches = cv2.drawMatchesKnn(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the image with keypoints and lines
cv2.imshow("Matches", image_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()
