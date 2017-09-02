from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
#from keras.models import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import metrics
from keras.optimizers import SGD
import os

import cv2
import numpy as np

import keras
from PIL import ImageFile
import PIL

samplePath = 'ayanami_rei'

filenames = []
for root, dirs, files in os.walk(samplePath):
    filenames = files

tensor = np.zeros((len(filenames),150,150,3))

for i in range(0,len(filenames)):
	curr_path = samplePath + '/' + filenames[i]
	#img = cv2.imread(curr_path)
	#img = PIL.Image.resize((150, 150))
	img = PIL.Image.open(curr_path)
	img = img.resize((150,150))
	tensor[i] = img

model = keras.models.load_model('100epoch_dratocos_35cls_64bat_softmax.h5')

'''
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
'''

print(tensor.shape)

x = model.predict_classes(tensor)
#for i in range(0,len(filenames)):
#	print(x[i])
print(x)