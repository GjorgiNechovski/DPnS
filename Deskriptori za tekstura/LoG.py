import cv2
import numpy as np

# Load images
image1 = cv2.imread("flower1.png", 1)
image2 = cv2.imread("flower2.png", 1)

# Resize images (optional)
image1 = cv2.resize(image1, (800, 600))
image2 = cv2.resize(image2, (800, 600))

# Convert images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Apply LoG filter
log1 = cv2.Laplacian(gray1, cv2.CV_64F)
log2 = cv2.Laplacian(gray2, cv2.CV_64F)

# Normalize the images
log1 = cv2.normalize(log1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
log2 = cv2.normalize(log2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Create a horizontal stack of the images
combined_image = np.hstack((log1, log2))

# Display the combined image
cv2.imshow("Combined LoG Images", combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
