import cv2
import mediapipe as mp
import time
import numpy as np
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

class VolumeHandControl():

    def __init__(self, pixelRange=[50, 300]):
        
        # Volume Control
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = interface.QueryInterface(IAudioEndpointVolume)
        volRange = self.volume.GetVolumeRange() # -65.25, 0.0, 0.03125
        self.minVol = volRange[0]
        self.maxVol = volRange[1]
        self.pixelRange = pixelRange

        self.volBar = np.interp(self.volume.GetMasterVolumeLevel(), [self.minVol, self.maxVol], [400, 150]) # to get current volume level at scale of 150-400
        self.volPer = np.interp(self.volume.GetMasterVolumeLevel(), [self.minVol, self.maxVol], [0, 100]) # to get current volume level at scale of 0-100

    def disp_volBar(self, img):
        '''
        Display volume bar on the screen
        '''
        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
        cv2.rectangle(img, (50, int(self.volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(self.volPer)} %', (40, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        return img

    def control(self, img, lmList):
        '''
        Control volume using index and thumb finger
        '''
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        length =  math.hypot(x2 - x1, y2 - y1) # Euclidean distance

        # Hand range 50 - 300 to respective ranges
        vol = np.interp(length, self.pixelRange, [self.minVol, self.maxVol])
        self.volBar = np.interp(length, self.pixelRange, [400, 150])
        self.volPer = np.interp(length, self.pixelRange, [0, 100])

        self.volume.SetMasterVolumeLevel(vol, None) # set volume

        if length < 50:
            cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        
        if length > 300:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

        return img

    def get_vol(self):
        '''
        Get current volume level
        '''
        return self.volPer
