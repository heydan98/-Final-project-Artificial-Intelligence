import cv2
import imutils
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


labels = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F',
'G','H','I','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z']

model = load_model('/home/heydan/AI20/artificial_intelligent/test_lai/model.hdf5')

def sort_contours(cnts):
    BoundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, BoundingBoxes) = zip(*sorted(zip(cnts, BoundingBoxes),
        key=lambda b:b[1][0], reverse=False))
    return (cnts,BoundingBoxes)

class SegmentChar:
    def __init__(self) -> None:
        self.plate = None
        self.listChar = None
        self.thresh = None


    def loadplate(self,plate):
        plate = cv2.resize(plate,(600,400))
        self.plate = plate

    def showplate(self):
        cv2.imshow('Plate',self.plate)
        cv2.waitKey(0)

    def segmentPlate(self):
        # Process license plate images and find contours
        gray = cv2.cvtColor(self.plate,cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray,30,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # Arrange contours and bodingBoxes in left-to-right order
        cnts,BoundingBoxes = sort_contours(cnts)
        BoundingBoxesChar =[]

        # Find the matching bouding boxes and calculate the total height of the bouding boxes
        total_h = 0
        for b in BoundingBoxes:
            (x,y,w,h) = b
            if w > 10 and w< 200 and h >50 and h <300:
                BoundingBoxesChar.append(b)
                total_h+=h

        # If the license plate cannot be cut, return False
        if len(BoundingBoxes) == 0 or len(BoundingBoxesChar) == 0:
            return False
        
        # Calculate the average height of all bounding boxes
        heigh_char = total_h / len(BoundingBoxesChar)
        BoxChar = []



        # Filter out all boding Boxes of average height
        # Purpose of filtering noise elements inside the character
        for c in BoundingBoxesChar:
            if c[3] > heigh_char-5:
                BoxChar.append(c)
        
        # Returns True if the license plate is cut successfully
        self.listChar = BoxChar
        self.thresh = thresh
        return True
    
    def ReadCharPlate(self):
        numberPlate = []
        for box in self.listChar:
            (x,y,w,h) = box
            cropped = self.thresh[y-5:min(y+h+5,self.thresh.shape[0]),max(x-12,0):min(x+w+13,self.thresh.shape[1])]
            try:
                cropped = cv2.resize(cropped,(32,32))
                cropped = img_to_array(cropped)
                boxchar = np.expand_dims(cropped,axis=0)
                pred = model.predict(boxchar).argmax(axis=1)
                char = labels[pred[0]]
                numberPlate.append(char)
                cv2.putText(self.plate,char,(x,y-5),cv2.FONT_HERSHEY_COMPLEX,0.75,(0,0,255),2)
                cv2.rectangle(self.plate,(x,y),(x+w,y+h),(0,255,0),3)
            except:
                continue
        return numberPlate

    