from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend

class LeNet:
    @staticmethod
    # Achitecture Lenet: INPUT => CONV => ACT => POOL => CONV => ACT => POOL => FC => ACT => FC
    def build(width,height,depth,classes):
        model = Sequential()
        inputShape = (height,width,depth)

        if backend.image_data_format() == 'channels_first':
            inputShape = (depth,height,width)

        # first set of CONV  => RELU => POOL layers
        model.add(Conv2D(20,(5,5), padding='same', input_shape = inputShape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        #second set of CONV => RELU => POOL layers
        model.add(Conv2D(50,(5,5), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        #first set of FC => RELU
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))

        #softmax classifier
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model