#this script is used to preprocess the data for the ssd mobilenet v2 model 
#it resizes the images to 640x480 and adds padding to the images so the images don't get distorted
#it also resizes the bounding boxes to the new image size

#the images where labeled with the VoTT tool
#the labels are in the json format

#used data:
# Animalfaces: https://www.kaggle.com/andrewmvd/animal-faces
# CatsVsDogs: https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset
# some personal pictures of my dogs


path = r"C:\Users\viers\OneDrive\Bureaublad\ResearchProject\researchproject\ssd-mobilenet-v2\data\dogFacesFull"
target_path = r"C:\Users\viers\OneDrive\Bureaublad\ResearchProject\researchproject\ssd-mobilenet-v2\data\processed"

import os
import json
import cv2
import numpy as np


folders = os.listdir(path)
for folder in folders:
    images = os.listdir(path + '\\' + folder + '\\Images')
    print(f"Pictures in {folder}: {len(images)}")

print("")

#read the labels and saves the images and bounding boxes in lists
images = []
faces = []
for folder in folders:
    labels = os.listdir(path + '\\' + folder + '\\labels')
    print(f"Labeled pictures in {folder}: {len(labels)}")
    for label in labels: 
        if label.endswith(".json"):
            with open(path+'\\'+folder+'\\labels\\'+label) as f:
                data = json.load(f)
                image = data['asset']['path'].split(":")[-1].split("/")[-1]
                images.append(path + '\\' + folder + '\\Images\\' + image)
                boundingBoxes = ""
                for i in range(len(data['regions'])):            
                    if boundingBoxes == "":
                        boundingBoxes = [data['regions'][i]['boundingBox']]
                    else:
                        boundingBoxes.append(data['regions'][i]['boundingBox'])
                faces.append(boundingBoxes)
print("")
print(images[0])
print(faces[0])
print(len (images))
print(len (faces))
print(f"number of images with multiple faces: {len([i for i in faces if len(i) > 1])}")


img_size = (640,480)

#resize with padding
def resize_with_padding(img, size, boundingbox):
    #get the ratio of the new image to the old image
    ratio = min(size[0]/img.shape[1], size[1]/img.shape[0])
    #get the new size of the image
    new_size = (int(img.shape[1]*ratio), int(img.shape[0]*ratio))
    #get the padding
    delta_w = size[0] - new_size[0]
    delta_h = size[1] - new_size[1]
    #add padding
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    img = cv2.resize(img, new_size)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    #resize bounding box
    #check if multiple bounding boxes
    if len(boundingbox) > 1:
        for i in range(len(boundingbox)):
            boundingbox[i]['left'] = int(boundingbox[i]['left']*ratio) + left
            boundingbox[i]['top'] = int(boundingbox[i]['top']*ratio) + top
            boundingbox[i]['width'] = int(boundingbox[i]['width']*ratio)
            boundingbox[i]['height'] = int(boundingbox[i]['height']*ratio)
    else:
        boundingbox[0]['left'] = int(boundingbox[0]['left']*ratio) + left
        boundingbox[0]['top'] = int(boundingbox[0]['top']*ratio) + top
        boundingbox[0]['width'] = int(boundingbox[0]['width']*ratio)
        boundingbox[0]['height'] = int(boundingbox[0]['height']*ratio)
    return img, boundingbox


   
# resize images and bounding boxes and save them to the target folder
i = 0
for image in images:

    img = cv2.imread(image)
    img, faces[i] = resize_with_padding(img, img_size, faces[i])
    cv2.imwrite(target_path + '\\Images\\' + str(i) + '.jpg', img)
    #write bounding boxes to txt file
    with open(target_path + '\\labels\\' + str(i) + '.txt', 'w') as f:
        for boundingBox in faces[i]:
            f.write(f"0 {boundingBox['left']} {boundingBox['top']} {boundingBox['width']} {boundingBox['height']} ")
    i += 1
