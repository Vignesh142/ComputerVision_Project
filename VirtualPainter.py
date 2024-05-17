import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

###############################################
brushThickness = 15
eraserThickness = 100
xp, yp = 0, 0
folderpath = "Painter_Menu"
drawColor = (255, 255, 255)
instructions = "Choose the menu with Middle & Index. Draw with Index finger"
imgCanvas = np.zeros((720,1280, 3), np.uint8)
###############################################

myList = os.listdir(folderpath)
# print(myList)

overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderpath}/{imPath}')
    image = cv2.resize(image, (1280, 125))
    overlayList.append(image)
# print(len(overlayList))

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.HandDetector(detectionCon=0.85)

header = overlayList[5]

########################################################
# Util functions
def clear_button(img):
    img = cv2.rectangle(img, (1150, 135), (1280, 170), (0, 255, 0), cv2.FILLED)
    img = cv2.putText(img, "Clear", (1155, 160), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)
    return img

def select_header(x1, y1):
    global header, overlayList, drawColor
    # print("Selection Mode")
    if y1 < 125:
        if 250 < x1 < 350:
            header = overlayList[0]
            drawColor = (0, 0, 255)
        elif 450 < x1 < 550:
            header = overlayList[1]
            drawColor = (0, 255, 0)
        elif 650 < x1 < 750:
            header = overlayList[2]
            drawColor = (255, 0, 255)
        elif 850 < x1 < 950:
            header = overlayList[3]
            drawColor = (255, 0, 0)
        elif 1050 < x1 < 1150:
            header = overlayList[4]
            drawColor = (0, 0, 0)
        else:
            header = overlayList[5]
    return header

def compute(img):
    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0) # Orr
    global imgCanvas
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)
    return img

###############################################

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)


    # Find Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if drawColor == (255, 255, 255):   
        cv2.putText(img, instructions, (50, 300), cv2.FONT_ITALIC, 1, (255, 125, 125), 5)
    
    if len(lmList) != 0:
        # print(lmList)

        # tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

        # If Selection Mode - Two finger are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            cv2.rectangle(img, (x1, y1-25), (x2, y2+25), drawColor, cv2.FILLED)
            header = select_header(x1, y1)
            img[0:125, 0:1280] = header
            if x1>1150 and y1>135 and x1<1280 and y1<170: # Clear button
                imgCanvas = np.zeros((720, 1280, 3), np.uint8)
                
        # If Drawing Mode - Index finger is up
        if fingers[1] and fingers[2] == False and drawColor != (255, 255, 255):
            # print("Drawing Mode")
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)

            if xp==0 and yp==0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1
        if fingers[1]==False and fingers[2]==False:
            xp, yp = 0, 0
    else:
        xp, yp = 0, 0

    img = compute(img)

    # Setting the menu bar
    img[0:125, 0:1280] = header
    # Clear button
    img = clear_button(img)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
