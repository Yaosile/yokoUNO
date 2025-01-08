import myOwnLibrary as myJazz
import numpy as np

import cv2 as cv2
import MelMyBoy as audio
import time

cap = cv2.VideoCapture(0)
time.sleep(5)
card = np.zeros((100,100))
if cap.isOpened():
    try:
        while True:
            ret, frame = cap.read()
            cv2.imshow('Webcam', frame.astype(np.uint8))
            cv2.imshow('Card', card.astype(np.uint8))
            key = cv2.waitKey(1) & 0xFF 
            if key == ord('q'):
                break
            elif key == ord(' '):
                card = frame.copy()
                lab = cv2.cvtColor(card, cv2.COLOR_BGR2LAB)
                # Split LAB channels
                l, a, b = cv2.split(lab)
                # Apply CLAHE to the L channel
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                # Merge channels and convert back to BGR color space
                limg = cv2.merge((cl, a, b))
                card = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
                card = card - card.min()
                card = card/card.max()
                card = card * 255
                card = myJazz.isolateCard(card)
                value = myJazz.isolateValue(card)
                value = myJazz.compareTemplate(value)
                print(myJazz.guessIndex[value])
    finally:
        print('Failed')
        cap.release()
        cv2.destroyAllWindows()