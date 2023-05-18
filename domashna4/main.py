import os
import cv2

def get_sift_descriptors(img):
    image = cv2.imread(img)
    greyscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(greyscale_image, None)

    return descriptors


def main():
    similarities = {}
    input_folder = './database/'
    query_folder = './query_images/'

    query_descriptors = get_sift_descriptors(os.path.join(query_folder, input('Image: ')))
    images = [os.path.join(input_folder, i)
              for i in os.listdir(input_folder)
              if os.path.isfile(os.path.join(input_folder, i))]

    for image in images:
        descriptors = get_sift_descriptors(image)

        similarity = cv2.matchShapes(query_descriptors, descriptors, 1, 0)

        similarities[os.path.basename(image)] = similarity

    for k, v in sorted(similarities.items(), key=lambda x: x[1], reverse=True):
        print(f'{k}:\t{v:.2f}')


if __name__ == '__main__':
    main()

#go koristam SIFT algoritmot za detekcija na kluchni tochki
#otkako ke se pronajdat dvata deskriptori za slikite shto se sporeduvaat istite odat vo funkcija matchShapes()
#taa vrakja vrednost od 1 do 0, kade 1 slikite se isti, a kolku e pomala vrednosta tolku se porazlichni slikite
#rezultatite se sortirani po opagjachki redosled