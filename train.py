from keras.optimizers import Adam
from models import basic, nv, test
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
number_of_epochs = 11
batch_size = 64
number_of_samples_per_epoch = 20032
validation_split = 0.3

# construct the selected model and print it to the screen
model = basic()
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
print("Total samples: ", len(csv))
print("Training size: ", len(train_data))
print("Validation size: ", number_of_validation_samples)

train_gen = generate_next_batch(train_data)
validation_gen = generate_next_batch(val_data)

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
