
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
offset = 20
imgSize = 300
counter = 0

folder = "Data\Love"

while True:
    success, img = cap.read()
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

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
