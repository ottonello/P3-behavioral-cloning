import errno
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Some useful constants
DRIVING_LOG_FILE = 'driving_log.csv'
# Taken from https://github.com/upul/behavioral_cloning
STEERING_COEFFICIENT = 0.229
COLUMNS = ['center','left','right','steering','throttle','brake','speed']

# TODO: actual augmentation

# center,left,right,steering,throttle,brake,speed
def get_random_camera_data(csv, index):
    """Get one of the left, center or right images together with
    the corresponding(adjusted) steering angle.
    """
    rnd = np.random.randint(0, 3)
    img = csv.iloc[index][COLUMNS.index('center') + rnd].strip()
    angle = csv.iloc[index][COLUMNS.index('steering')]
    
    # Adjust steering based on camera position
    if COLUMNS.index('center') + rnd == COLUMNS.index('left'):
        angle = angle + STEERING_COEFFICIENT
    elif rnd == COLUMNS.index('right'):
        angle = angle - STEERING_COEFFICIENT

    return (img, angle)


def next_batch(base_dir, batch_size=64):
    log_file = os.path.join(base_dir, DRIVING_LOG_FILE)
    csv = pd.read_csv(log_file)
    # Get a random batch of data rows
    random_rows = np.random.randint(0, len(csv), batch_size)
    
    batch = []
    for index in random_rows:
        data = get_random_camera_data(csv, index)
        batch.append(data)

    return batch


def data_generator(base_dir, batch_size=64):
    while True:
        X_batch = []
        y_batch = []
        images = next_batch(base_dir, batch_size)
        for img_file, angle in images:
            img_file = os.path.join(base_dir, img_file)
            raw_image = plt.imread(img_file)
            raw_angle = angle
            X_batch.append(raw_image)
            y_batch.append(raw_angle)

        assert len(X_batch) == batch_size, 'len(X_batch) == batch_size should be True'

        yield np.array(X_batch), np.array(y_batch)
