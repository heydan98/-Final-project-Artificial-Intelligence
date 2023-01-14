from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import glob

import re
import cv2
import argparse
import imutils
import numpy as np
from crop_licence_plate import LicencePlate
from segment_letter_lp import SegmentChar

app = Flask(__name__)

UPLOAD_FOLDER = '/home/heydan/AI20/artificial_intelligent/test_lai/static/uploads'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
 

lp = LicencePlate()
def Detect_Plate(uploaded_image_path):
    image = cv2.imread(uploaded_image_path)
    height, width, channels = image.shape
    image = cv2.resize(image,(620,480))
    lp.load_image(uploaded_image_path)
    plate,cnt = lp.crop_plate()
    result = 'Unable to recognize license plate'
    if plate is not None:

        segchar = SegmentChar()
        segchar.loadplate(plate)
        
        suc = segchar.segmentPlate()
        if suc:
            numPlate = segchar.ReadCharPlate()
            #segchar.showplate()
            result = 'License plate: '+ ''.join(numPlate)
    if result != 'Unable to recognize license plate':
        try:
            cv2.drawContours(image, [cnt], -1, (0,255,0), 3)
        except:
            print("An exception occurred")
    image = cv2.resize(image,(width,height))
    Clear_Upload_Folder()
    cv2.imwrite(uploaded_image_path, image)
    return result

def Clear_Upload_Folder():
    files = glob.glob('static/uploads/*')
    for f in files:
        os.remove(f)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash("Haven't selected a photo yet")
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        KetQua=Detect_Plate(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash(KetQua)

        return render_template('index.html', filename=filename)
    else:
        flash('Photo must be in one of the formats :  png, jpg, jpeg')
        return redirect(request.url)

 
@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 9600))
    app.run(host='0.0.0.0', port=port, debug=True)