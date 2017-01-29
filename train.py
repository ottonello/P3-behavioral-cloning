from keras.optimizers import Adam
from models import basic, nv
from generators import data_generator
import json

learning_rate = 1e-4
number_of_epochs = 20
batch_size = 256
number_of_samples_per_epoch = 25600
number_of_validation_samples = 256

OUTPUT_MODEL_FILE = "model.json"
OUTPUT_WEIGHTS_FILE = "model.h5"

DATA_DIR = "udacity"

print("Dataset:", DATA_DIR)

# construct the selected model
model = nv()
model.summary()
model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate))

# create two generators for training and validation
train_gen = data_generator(DATA_DIR, batch_size)
validation_gen = data_generator(DATA_DIR, batch_size)

history = model.fit_generator(train_gen,
                  samples_per_epoch=number_of_samples_per_epoch,
                  nb_epoch=number_of_epochs,
                  validation_data=validation_gen,
                  nb_val_samples=number_of_validation_samples,
                  verbose=1)

with open(OUTPUT_MODEL_FILE, 'w') as outfile:
    json.dump(model.to_json(), outfile)
model.save_weights(OUTPUT_WEIGHTS_FILE)