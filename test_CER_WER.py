import re
import cv2
import argparse
import imutils
import numpy as np
from crop_licence_plate_test import LicencePlate
from segment_letter_lp import SegmentChar
from flask import Flask,render_template
import os
import pandas as pd
import fastwer

# def Test2():
labels=pd.read_excel('/home/heydan/AI20/artificial_intelligent/test_lai/test_dataset/labels.xlsx')
labels['ID']=labels['ID'].map(str)
file_list=os.listdir(r"/home/heydan/AI20/artificial_intelligent/test_lai/test_dataset/images")
sum=0
cer_sum=0
wer_sum=0
countPlate=0
for path in file_list:
    no=path[:-4]
    row=labels['NUMBER'].where(labels['ID'] == no).dropna().values[0]
    image_path = '/home/heydan/AI20/artificial_intelligent/test_lai/test_dataset/images/'+path
    lp = LicencePlate()
    # image = cv2.resize(image,(620,480))
    lp.load_image(image_path)
    plate, cnt = lp.crop_plate()
    result =''
    if plate is not None:
        segchar = SegmentChar()
        segchar.loadplate(plate)
        suc = segchar.segmentPlate()
        if suc:
            numPlate = segchar.ReadCharPlate()
            result = ''.join(numPlate)
            if len(result) >= 4:
                countPlate+=1
                cer = fastwer.score_sent(result, row , char_level=True)
                wer = fastwer.score_sent(result,row  , char_level=False)
                cer_sum=cer_sum+cer
                wer_sum= wer_sum + wer
        sum=sum+1

print(countPlate)
print(f"CER : {cer_sum/sum} %")
print(f"WER : {wer_sum/sum} %")


# Test2()