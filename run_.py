import re
import cv2
import argparse
import imutils
import numpy as np
from crop_licence_plate_test import LicencePlate
from segment_letter_lp import SegmentChar
from keras.models import load_model

model =load_model("model.hdf5")

ap = argparse.ArgumentParser()
ap.add_argument('-i','--image',required=True)
args = vars(ap.parse_args())

lp = LicencePlate()
lp.load_image(args['image'])
image = lp.image
plate = lp.crop_plate()

# plate = cv2.imread('/home/heydan/AI20/artificial_intelligent/test_lai/bien.png')
result = "Unable to recognize license plate"

if plate is not None:
    segchar = SegmentChar()
    segchar.loadplate(plate)
    # cv2.imshow('Car', image)
    
    suc = segchar.segmentPlate()
    if suc:
        numPlate = segchar.ReadCharPlate()
        segchar.showplate()
        result = 'License plate: '+ ''.join(numPlate)

cv2.putText(image, result,(20,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
cv2.imshow('Car', image)
cv2.waitKey(0)
print(result)
cv2.destroyAllWindows()