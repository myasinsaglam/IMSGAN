from util import DataLoader

import numpy as np

import imageio
import cv2

# loader = DataLoader("/Users/Macbook/Desktop/Dataset/256/256",256,256,64,64,4)
# indexes = np.arange(1,10)

# np.random.shuffle(indexes)

# print(indexes)
#
# img = cv2.imread("/Users/Macbook/Desktop/FinalYearProject/IMSGAN/SRGAN-Keras-master/data/29.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img2 =imageio.imread("/Users/Macbook/Desktop/FinalYearProject/IMSGAN/SRGAN-Keras-master/data/29.jpg").astype(np.float)
#
# print(type(img),type(img2))
#
# print(img[0],img2[0])



from keras.models import Sequential, Model,load_model
from keras.layers import Input, Activation, Add
from keras.layers import BatchNormalization, LeakyReLU, Conv2D, Dense
from keras.layers import UpSampling2D
from keras.optimizers import Adam
from keras.applications import VGG19

from keras.callbacks import TensorBoard, ReduceLROnPlateau


vgg19 = VGG19()
# vgg19.summary()
# print((vgg19.layers[9].name))

from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt

def build_vgg_face(weights_path):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1),input_shape=(224,224,3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    model.load_weights(weights_path)
    model.summary()
    return model



# print(vgg.layers[15].name)

# Input image to extract features from
img = Input(shape=(256,256,3))

# Get the vgg network. Extract features from last conv layer
vgg = VGG19(weights="imagenet")
# vgg = build_vgg_face("/Users/Macbook/Desktop/FinalYearProject/IMSGAN/vgg_face_weights.h5")
vgg.outputs = [vgg.layers[9].output]

# Create model and compile
model = Model(inputs=img, outputs=vgg(img))

model.summary()