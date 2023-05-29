import cv2

# Load images
image1 = cv2.imread("discord1.png")
image2 = cv2.imread("discord2.png")

# Resize images (optional)
image1 = cv2.resize(image1, (800, 600))
image2 = cv2.resize(image2, (800, 600))

# Apply Gaussian blur
image1 = cv2.GaussianBlur(image1, (5, 5), 0)
image2 = cv2.GaussianBlur(image2, (5, 5), 0)

# Create ORB object
orb = cv2.ORB_create()

# Detect keypoints and compute descriptors
keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

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

similarity_formatted = "{:.5f}".format(similarity)

# Print similarity score
print("Similarity score:", similarity_formatted)

# Draw matches between images
image_matches = cv2.drawMatchesKnn(
    image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# Display image with matches
cv2.imshow("Matches", image_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()
