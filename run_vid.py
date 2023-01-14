import cv2 
import re
import cv2
import argparse
import imutils
import numpy as np
import os
from segment_letter_lp_vid import SegmentChar
from keras.models import load_model
from tkinter import filedialog

model =load_model("model.hdf5")

ap = argparse.ArgumentParser()
# ap.add_argument('-i','--image',required=True)
args = vars(ap.parse_args())

# Create a video capture object, in this case we are reading the video from a file
file = filedialog.askopenfilename(initialdir= 'TEST_LAI')

vid_capture = cv2.VideoCapture(file)
# vid_capture = cv2.VideoCapture('/home/heydan/AI20/artificial_intelligent/test_lai/test.mp4')

if (vid_capture.isOpened() == False):
	print("Error opening the video file")
# Read fps and frame count
else:
	# Get frame rate information
	# You can replace 5 with CAP_PROP_FPS as well, they are enumerations
	fps = vid_capture.get(5)
	print('Frames per second : ', fps,'FPS')

	# Get frame count
	# You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
	frame_count = vid_capture.get(7)
	print('Frame count : ', frame_count)

while(vid_capture.isOpened()):
    # vid_capture.read() methods returns a tuple, first element is a bool 
    # and the second is frame
    if model is not None:
        success, image1 = vid_capture.read()
        img = image1[200:900, 0:600]
        image = cv2.resize(img, dsize=(620, 480))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 10, 15, 15)
        edged = cv2.Canny(blurred,30,200)
        cnts = None
        cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if cnts is not None:
            cnts = imutils.grab_contours(cnts)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
            screenCnt = None

            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c,0.02*peri, True)
                area = cv2.contourArea(c)

                if len(approx) == 4 and (10000>area > 7000 or area>14000) :
                    screenCnt = approx
                    continue
        
            if screenCnt is not None:
                mask = np.zeros_like(gray, np.uint8)
                cv2.drawContours(mask,[screenCnt],0,255,-1)

                (x,y) = np.where(mask == 255)
                (topX, topY) = (np.min(x), np.min(y))
                (botX, botY) = (np.max(x), np.max(y))

                plate = image[topX:botX+1, topY: botY+1]
                # cv2.imshow('test',plate)
                # print(plate.shape)
                
                # result = 'Khong the nhan dang'
                if plate.shape[1]>100: #is not None:
                    # cv2.imwrite('bien.png', plate)
                    # cv2.imshow("test", plate)
                    segchar = SegmentChar()
                    segchar.loadplate(plate)
                    # print(plate.shape)    
                    suc = segchar.segmentPlate()
                    # cv2.imshow("suc", suc)
                    if suc:
                        numPlate = segchar.ReadCharPlate()
                        # cv2.imshow("test", numPlate)
                        # segchar.showplate()
                        if len(numPlate)>4 :
                            result = 'Licence plate: '+ ''.join(numPlate)
                            print(numPlate)
                            cv2.putText(image1, result,(20,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
                            path = "/home/heydan/AI20/artificial_intelligent/test_lai/image_from_video"
                            cv2.imwrite(os.path.join(path ,"Lp"+str(numPlate).replace(",","'")+'.png'), segchar.plate)
                            cv2.imwrite(os.path.join(path ,"Car"+str(numPlate).replace(",","'")+'.png'), image1)

                # cv2.putText(image, result,(20,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
        cv2.imshow("test", image1)
        # cv2.imshow("test", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        break

# Release the video capture object
vid_capture.release()
cv2.destroyAllWindows()