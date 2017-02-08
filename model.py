from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.layers.core import Lambda
from keras.layers.advanced_activations import ELU
from generators import generate_next_batch
from keras import backend as K
import pandas as pd
import numpy as np

# Input files
DATA_PATH='./udacity'
DRIVING_LOG_FILE = DATA_PATH + '/driving_log.csv'

# Output files
OUTPUT_MODEL_FILE = "model.json"
OUTPUT_WEIGHTS_FILE = "model.h5"

# Training parameters
learning_rate = 1e-4
number_of_epochs = 7
batch_size = 64
number_of_samples_per_epoch = 20032
validation_split = 0.3

# Images size
resize_x=64
resize_y=64

# Input layer shape
ch, row, col = 3, resize_x, resize_y

def nv():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(col,row,ch),
            output_shape=(col,row,ch)))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(64, 3, 3, border_mode="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(64, 3, 3, border_mode="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(Activation('relu'))
    model.add(Dense(1164))
    model.add(Dropout(.5))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Dropout(.5))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Dropout(.5))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Dropout(.5))
    model.add(Activation('relu'))
    model.add(Dense(1))
    return model

# construct the selected model and print it to the screen
model = nv()
model.summary()
model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate))

def split(csv, val_split):
	shuffled = csv.iloc[np.random.permutation(len(csv))]
	validation_samples = int(len(csv) * val_split)
	return (shuffled[validation_samples:],
				shuffled[:validation_samples])

# Split samples into training and validation
csv = pd.read_csv(DRIVING_LOG_FILE)
train_data, val_data = split(csv, validation_split)
number_of_validation_samples = len(val_data)

print("Log File:", DRIVING_LOG_FILE)
print("Total samples: ", len(csv))
print("Training size: ", len(train_data))
print("Validation size: ", number_of_validation_samples)

train_gen = generate_next_batch(train_data, resize_dim=(resize_x, resize_y))
validation_gen = generate_next_batch(val_data, resize_dim=(resize_x, resize_y))

history = model.fit_generator(train_gen,
                  samples_per_epoch=number_of_samples_per_epoch,
                  nb_epoch=number_of_epochs,
                  validation_data=validation_gen,
                  nb_val_samples=number_of_validation_samples,
                  verbose=1)

model_json = model.to_json()
with open(OUTPUT_MODEL_FILE, "w") as json_file:
    json_file.write(model_json)
model.save(OUTPUT_WEIGHTS_FILE)

K.clear_session()



###
## Other Model definitions
###


def basic():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(col,row,ch),
            output_shape=(col,row,ch)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 5, 5, border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode="same"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(Activation('relu'))
    model.add(Dense(400))
    model.add(Dropout(.5))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Dropout(.5))
    model.add(Activation('relu'))
    model.add(Dense(20))
    model.add(Dropout(.5))
    model.add(Activation('relu'))
    model.add(Dense(1))

    return model
