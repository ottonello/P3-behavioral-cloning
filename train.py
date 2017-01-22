from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.optimizers import Adam
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.layers.core import Lambda
from keras.layers.advanced_activations import ELU
from scipy import ndimage
from PIL import Image

import numpy as np
import csv
import sys
import os

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

print("Proportion of nonzero values: %s" % (np.count_nonzero(y_train) / len(y_train)))

X_train = []
for imageName in X_input:
    X_train.append(ndimage.imread(imageName))

# img = Image.fromarray(X_train[0])
# img.show()

X_train = np.array(X_train)

ch, row, col = 3, 320, 160

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

# model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), metrics=['accuracy'])

model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])
# model.compile('adam', 'mean_squared_error')

model.fit(X_train, y_train, nb_epoch=20, batch_size=128,validation_split=0.2)

# print(history.history['acc'][-1])
#loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)
model.save('model.h5')

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

import gc; gc.collect()