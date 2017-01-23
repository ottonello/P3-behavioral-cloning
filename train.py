from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.layers.core import Lambda
from keras.layers.advanced_activations import ELU
from scipy import ndimage
from PIL import Image

import numpy as np
import csv
import sys
import os

ch, row, col = 3, 320, 160

def basic():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(col,row,ch),
            output_shape=(col,row,ch)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default'))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, border_mode="same"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default'))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    return model

def nv():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(col,row,ch),
            output_shape=(col,row,ch)))
    model.add(Convolution2D(24, 3, 3, subsample=(2, 2), border_mode="valid"))
    model.add(ELU())
    model.add(Convolution2D(36, 3, 3, subsample=(2, 2), border_mode="valid"))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, border_mode="valid"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, border_mode="valid"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, border_mode="valid"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(100))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(50))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(10))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    return model

args = len(sys.argv)
if args < 2 :
	print("Provide the data directory")
	quit()

basedir = sys.argv[1]
log_file = os.path.join(basedir, 'driving_log.csv')

with open(log_file, 'r') as csvfile:
	reader = csv.reader(csvfile)
	data = np.array(list(reader))
	
# Load data
X_input = data[:,0]
y_train = np.array(data[:,3], dtype="float32")

print("Dataset:", basedir)
print("Proportion of nonzero values:", (np.count_nonzero(y_train) / len(y_train)))

X_train = []
for imageName in X_input:
    X_train.append(ndimage.imread(imageName))

# img = Image.fromarray(X_train[0])
# img.show()

X_train = np.array(X_train)

datagen = ImageDataGenerator(horizontal_flip=False)
#datagen.fit(X_train)

model = basic()
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0))

model.fit_generator(datagen.flow(X_train, y_train, batch_size=256), samples_per_epoch=len(X_train), nb_epoch=10)

# print(history.history['acc'][-1])
#loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)
model.save('model.h5')

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

import gc; gc.collect()
