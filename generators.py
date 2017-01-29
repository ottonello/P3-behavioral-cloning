import errno
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Some useful constants
DRIVING_LOG_FILE = 'driving_log.csv'
# Taken from https://github.com/upul/behavioral_cloning
# STEERING_COEFFICIENT = 0.229
offset=1.0 
dist=20.0
dsteering = offset/dist * 360/( 2*np.pi) / 25.0

COLUMNS = ['center','left','right','steering','throttle','brake','speed']

RESIZE_W = 64
RESIZE_H = 64

# center,left,right,steering,throttle,brake,speed
def get_random_camera_data(csv, index):
    """Get one of the left, center or right images together with
    the corresponding(adjusted) steering angle.
    """
    rnd = np.random.randint(0, 3)
    img = csv.iloc[index][COLUMNS.index('center') + rnd].strip()
    angle = csv.iloc[index][COLUMNS.index('steering')]
    
    # Adjust steering based on camera position
    if rnd == COLUMNS.index('left'):
        angle = angle + dsteering
    elif rnd == COLUMNS.index('right'):
        angle = angle - dsteering

    return (img, angle)

def resize(image):
    return cv2.resize(image, dsize=(RESIZE_W, RESIZE_H))

def random_flip(image, steering_angle, flip_probability = 0.5):
    coin_toss = np.random.choice(2, p=[1-flip_probability, flip_probability])
    if coin_toss:
        flipped = cv2.flip(image, 1)
        return (flipped,  (-1) * steering_angle)
    else:
        return (image,  steering_angle)

def preprocess(image, steering_angle):
    image = resize(image)
    image, steering_angle = random_flip(image, steering_angle)
    return (image, steering_angle)

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
            processed_image, processed_angle = preprocess(raw_image, raw_angle)
            X_batch.append(processed_image)
            y_batch.append(processed_angle)

        assert len(X_batch) == batch_size, 'len(X_batch) == batch_size should be True'

        yield np.array(X_batch), np.array(y_batch)
