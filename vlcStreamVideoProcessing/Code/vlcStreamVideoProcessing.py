import numpy as np
import pytesseract as tess
import urllib
import cv2
#import matplotlib.pyplot as plt

import ga

import os
import time

import vlc

#instance = vlc.Instance()
#mediaplayer = instance.media_player_new("/home/sd/Videos/081118.mp4")
#mediaplayer.play()

#instance2 = vlc.Instance()
#mediaplayer2 = instance2.media_player_new("/home/sd/Videos/081118.mp4")
#mediaplayer2.play()

instance3 = vlc.Instance()
mediaplayer3 = instance3.media_player_new("rtsp://127.0.0.1:8554/")
mediaplayer3.play()

while 1:
    time.sleep(1)
    mediaplayer3.video_take_snapshot(0, '/home/sd/Videos/snapshotTemp.png', 0, 0)

    gray = cv2.imread("/home/sd/Videos/snapshotTemp.png", 0)

    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray, None)

    img3 = gray
    img2 = cv2.drawKeypoints(gray, kp, img3)

    cv2.imwrite('/home/sd/Videos/sift_keypoints.jpg', img2)

    cv2.imshow('lalala', img2)

    dummy = 0



#vcap = cv2.VideoCapture(0)
vcap = cv2.VideoCapture("rtsp://127.0.0.1:8554/")
#vcap = cv2.VideoCapture("rtsp://127.0.0.1:8554")
#vcap = cv2.VideoCapture("rtsp://127.0.0.1:8554/out.h264")
#vcap = cv.VideoCapture("rtsp://192.168.1.2:8080/out.h264")


while(1):
    if (vcap.isOpened()):
        ret, frame = vcap.read()
        dummy = 1




dummy = 0

ver = tess.get_tesseract_version

x = np.array([1, 2])

print("Hello World!")

dummy = 0

x = np.array([0, 1, 2, 1, 4, 1, 5, 1, 3])


req = urllib.request.urlopen('http://answers.opencv.org/upfiles/logo_2.png')
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
img = cv2.imdecode(arr, -1) # 'Load it as it is'

gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)

img3=gray
img2=cv2.drawKeypoints(gray, kp, img3)

cv2.imwrite('sift_keypoints.jpg',img2)

cv2.imshow('lalala', img2)
if cv2.waitKey() & 0xff == 27: quit()

#plt.show()
#plt.ion()

#plt.imshow(img2)

cv2.imshow('lalala', dst)
if cv2.waitKey() & 0xff == 27: quit()
dst = cv2.medianBlur(img3, 5)
dst = cv2.GaussianBlur(dst,(5,5),0)



dummy = 0






