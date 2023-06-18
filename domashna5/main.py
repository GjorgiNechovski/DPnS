import os
import cv2


def getImages(directory: str):
    files = os.listdir(directory)
    images = []

    for file in files:
        imagePath = os.path.join(directory, file)
        img = cv2.imread(imagePath, 1)
        img = cv2.resize(img, (800, 600)) #ako ne napravam resize laptopot mi crash i PyCharm mu pravi force kill na procesot
        images.append(img)

    return images


def findBestMatches(questionedImageDescriptors, descriptors):
    bf = cv2.BFMatcher()
    goodMatches = []

    for descriptor in descriptors:
        matches = bf.knnMatch(questionedImageDescriptors, descriptor, k=2)

        good = []

        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])

        goodMatches.append(good)

    return goodMatches


def drawMatchingImage(questionedImage, bestMatchImage, questionedImageKeypoints, bestMatchImageKeypoints, matches):
    matchingImage = cv2.drawMatches(
        questionedImage, questionedImageKeypoints,
        bestMatchImage, bestMatchImageKeypoints,
        matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return matchingImage


def main():
    questionedImage = cv2.imread(input("Enter the image filename: "), 1)
    images = getImages("Database")

    sift = cv2.SIFT_create()

    questionedImageKeypoints, questionedImageDescriptors = sift.detectAndCompute(questionedImage, None)

    descriptors = []

    for image in images:
        keypoints, descriptor = sift.detectAndCompute(image, None)
        descriptors.append(descriptor)

    matches = findBestMatches(questionedImageDescriptors, descriptors)

    bestMatchIndex = 0
    bestMatchCount = len(matches[0])

    for i in range(1, len(matches)):
        if len(matches[i]) > bestMatchCount:
            bestMatchIndex = i
            bestMatchCount = len(matches[i])

    bestMatchImage = cv2.resize(images[bestMatchIndex], (questionedImage.shape[1], questionedImage.shape[0]))

    bestMatchImageKeypoints = sift.detect(bestMatchImage, None)
    bestMatchImageWithKeypoints = cv2.drawKeypoints(bestMatchImage, bestMatchImageKeypoints, None)

    questionedImageWithKeypoints = cv2.drawKeypoints(questionedImage, questionedImageKeypoints, None)

    matches = [match[0] for match in matches[bestMatchIndex] if match]

    matchingImage = drawMatchingImage(questionedImage, bestMatchImage, questionedImageKeypoints,
                                      bestMatchImageKeypoints, matches)

    cv2.imshow("Matching", matchingImage)
    cv2.imwrite("MatchingComparison.jpg", matchingImage)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
