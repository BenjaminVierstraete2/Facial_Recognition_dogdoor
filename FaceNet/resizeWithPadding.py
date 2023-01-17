path = "C:\\Users\\viers\\OneDrive\\Bureaublad\\ResearchProject\\researchproject\\FaceNet\\data\\pups\\"
targetPath = "C:\\Users\\viers\\OneDrive\\Bureaublad\\ResearchProject\\researchproject\\FaceNet\\data\\Images\\"

import os

images = os.listdir(path)
print(len(images))

import cv2
import numpy as np

imgSize = 1028

#resize with padding
def resizeWithPadding(img, size):
    height = img.shape[0]
    width = img.shape[1]
    if height > width:
        ratio = size / height
        height = size
        width = int(width * ratio)
    else:
        ratio = size / width
        width = size
        height = int(height * ratio)
    img = cv2.resize(img, (width, height))
    top = (size - height) // 2
    bottom = size - height - top
    left = (size - width) // 2
    right = size - width - left
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img

i = 0
for image in images:
    i += 1
    print(f'Image {i} of {len(images)}')
    img = cv2.imread(path + image)
    img = resizeWithPadding(img, imgSize)
    cv2.imwrite(targetPath + str(i) + '.jpg', img)


