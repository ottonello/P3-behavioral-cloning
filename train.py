from keras.optimizers import Adam
from models import basic, nv, test
from generators import generate_next_batch
from keras import backend as K

learning_rate = 1e-4
number_of_epochs = 11
batch_size = 64
number_of_samples_per_epoch = 20032
number_of_validation_samples = 640

OUTPUT_MODEL_FILE = "model.json"
OUTPUT_WEIGHTS_FILE = "model.h5"

# construct the selected model
model = basic()
model.summary()
model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate))

# create two generators for training and validation
train_gen = generate_next_batch()
validation_gen = generate_next_batch()

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
