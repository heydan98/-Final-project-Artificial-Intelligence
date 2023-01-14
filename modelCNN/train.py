from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import cv2
from lenet import LeNet
from imutils import paths
import numpy as np
from process import LoadPreprocess
from sklearn.metrics import classification_report

pathDS = '/home/heydan/AI20/artificial_intelligent/drive-download-20221216T190653Z-001/Car_letter_Dataset'
images = list(paths.list_images(pathDS))

lnp = LoadPreprocess(32,32)
x,y = lnp.load(images,2500)
x = x.astype('float') /255.0

train_val_X,testX,train_val_Y,testY = train_test_split(x,y,test_size=0.25,random_state=42)
trainX,valX,trainY,valY = train_test_split(train_val_X,train_val_Y)

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
valY = lb.fit_transform(valY)
testY = lb.fit_transform(testY)

model = LeNet.build(32,32,1,35)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics=['accuracy'])
filepath="best_model.hdf5"
from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]
H = model.fit(trainX,trainY, validation_data=(valX,valY), batch_size=256, epochs=20, verbose=1,  callbacks = callbacks_list)
model.save('model.hdf5')

predictions = model.predict(testX, batch_size=128)
print('Accuracy model: ',model.evaluate(testX,testY))
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1), 
                            target_names=[str(x) for x in lb.classes_]))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 20), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('plot.jpg')
plt.show()