from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
#from keras.models import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import metrics
from keras.optimizers import SGD

import cv2
import numpy as np

import keras
from PIL import ImageFile

img_width, img_height = 150, 150

img = cv2.imread('test12.jpg')
img = cv2.resize(img, (150, 150))
img = np.expand_dims(img, axis=0)

model = keras.models.load_model('100epoch_dratocos_35cls_64bat_softmax.h5')

'''
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
'''

x = model.predict_classes(img)
print(x)