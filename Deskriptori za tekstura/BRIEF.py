import cv2
import numpy as np

# Load images
image1 = cv2.imread('discord1.png', cv2.IMREAD_COLOR)
image2 = cv2.imread('discord2.png', cv2.IMREAD_COLOR)

image1 = cv2.resize(image1, (800, 600))
image2 = cv2.resize(image2, (800, 600))

# Convert images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Initialize FAST detector and BRIEF descriptor
fast = cv2.FastFeatureDetector_create()
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

# Detect keypoints with FAST
kp1 = fast.detect(gray1, None)
kp2 = fast.detect(gray2, None)

# Compute descriptors with BRIEF
kp1, des1 = brief.compute(gray1, kp1)
kp2, des2 = brief.compute(gray2, kp2)

# Match keypoints
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# Draw matched keypoints and lines
match_image = cv2.drawMatches(image1, kp1, image2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Calculate similarities
num_matches = len(matches)
similarity = num_matches / min(len(kp1), len(kp2))

# Print the similarity score
print("Similarity score:", similarity)

# Display the image with matches
cv2.imshow('Matches', match_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
