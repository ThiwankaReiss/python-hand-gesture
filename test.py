import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier = Classifier("C:/Thiwanka/ThiwankaPython/test rename/Handgesture/Models/keras_model.h5" , "C:/Thiwanka/ThiwankaPython/test rename/Handgesture/Models/labels.txt")
offset = 20
imgSize = 300
counter = 0

labels = ["Hi","I","Love","You","Rose for you"]

# Load small images
small_images = {
    "Hi": cv2.imread("C:/Thiwanka/ThiwankaPython/test rename/Handgesture/img/hi.png"),
    "I": cv2.imread("C:/Thiwanka/ThiwankaPython/test rename/Handgesture/img/I.png"),
    "Love": cv2.imread("C:/Thiwanka/ThiwankaPython/test rename/Handgesture/img/love.png"),
    "You": cv2.imread("C:/Thiwanka/ThiwankaPython/test rename/Handgesture/img/you.png"),
    "Rose for you": cv2.imread("C:/Thiwanka/ThiwankaPython/test rename/Handgesture/img/flower.png")
}

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        if len(hands)==1:
            hand = hands[0]
            x, y, w, h = hand['bbox']
        if len(hands)==2:
            hand1 = hands[0]
            x1, y1, w1, h1 = hand1['bbox']

            hand2 = hands[1]
            x2, y2, w2, h2 = hand2['bbox']

            if (x2>=x1):

                
                x=x1
                w=(x2-x1+w2)

            else:
                x=x2
                w=(x1-x2+w1)
                

            if (y2>=y1):

                y=y1
                h=(y2-y1+h2)

            else:
                y=y2
                h=(y1-y2+h1)


        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255

        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap: wCal + wGap] = imgResize
            prediction , index = classifier.getPrediction(imgWhite, draw= False)
            print(prediction, index)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize
            prediction , index = classifier.getPrediction(imgWhite, draw= False)

        label = labels[index]
        small_image = small_images[label]
        small_image_resized = cv2.resize(small_image, (50, 50))  # Resize to desired size

        # Position of the small image
        imgOutput[y-offset-60:y-offset-10, x:x+50] = small_image_resized
        
        cv2.putText(imgOutput, label, (x+60, y-30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,255,0), 2) 
        cv2.rectangle(imgOutput, (x-offset, y-offset), (x + w + offset, y + h + offset), (0, 0, 225), 2)   

        # cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', imgOutput)
    cv2.waitKey(1)
