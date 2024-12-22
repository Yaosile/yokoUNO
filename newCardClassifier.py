import numpy as np
from numpy import asanyarray as ana
import myOwnLibrary as myJazz
import cv2
from PIL import Image

def captureCard():
    boardFrame = 'Board'
    cardFrame = 'Card'
    card = np.ones((100,100,3))
    cx,cy = 0,0

    video_capture = cv2.VideoCapture(myJazz.gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    yuw, xuw = np.load('yMap.npy'), np.load('xMap.npy')
    if video_capture.isOpened():
        try:
            cv2.namedWindow(boardFrame, cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow(cardFrame, cv2.WINDOW_AUTOSIZE)
            while True:
                ret_val, frame = video_capture.read()
                frame = frame[yuw,xuw]
                if cv2.getWindowProperty(boardFrame, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(boardFrame,frame[::2,::2,:].astype(np.uint8))
                else:
                    break

                if cv2.getWindowProperty(cardFrame, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(cardFrame,card.astype(np.uint8))
                else:
                    break

                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    card = frame.copy()
                    card[card.shape[0]//2:,:] = 0
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
                    # card = myJazz.isolateCard(card)
                    # card, col = myJazz.getCardColour(card)
                    # print(col)
                    # if col not in ['Wild', 'Wild+4']:
                    #     print(col, myJazz.getCardValue(card))
                    # else:
                    #     print(col)
        finally:
            video_capture.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    captureCard()