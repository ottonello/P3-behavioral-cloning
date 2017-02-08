from generators import generate_next_batch
from PIL import Image
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import model_from_json
from keras.models import Model
import cv2

DATA_PATH='./udacity'
DRIVING_LOG_FILE = DATA_PATH + '/driving_log.csv'

def main():
	csv = pd.read_csv(DRIVING_LOG_FILE)
	
	gen = generate_next_batch(csv, batch_size=10, augment=False)
	imgs, angles = next(gen)

	with open('model.json', 'r') as jfile:
		model = model_from_json(jfile.read())

	model.compile("adam", "mse")
	weights_file = "model.h5"
	model.load_weights(weights_file)

	layer1 = Model(input=model.input, output=model.get_layer('convolution2d_1').output)
	layer2 = Model(input=model.input, output=model.get_layer('convolution2d_2').output)
	layer4 = Model(input=model.input, output=model.get_layer('convolution2d_4').output)

	show_images = True
	if show_images:
		for img, angle in zip(imgs,angles):
			print("angle: ", angle)
			print("original")
			temp = cv2.resize(img, (200, 66), cv2.INTER_AREA)
			plt.imshow(img)
			plt.show()
			img = np.expand_dims(img, axis=0)
			visual_layer1 = layer1.predict(img)
			visual_layer2 = layer2.predict(img)
			visual_layer4 = layer4.predict(img)

			print(np.shape(visual_layer1))
			arr_1 = np.swapaxes(visual_layer1[0], 2, 0)
			arr_2 = np.swapaxes(visual_layer2[0], 2, 0)
			arr_4 = np.swapaxes(visual_layer4[0], 2, 0)
			
			print("layer 1 feature map")

			plt.figure(figsize=(12,8))
			for i in range(24):
			    plt.subplot(8, 4, i+1)
			    temp = arr_1[i]
			    temp = cv2.resize(temp, (200, 66), cv2.INTER_AREA)
			    plt.imshow(temp)
			    plt.axis('off')
			plt.show()

			print("layer 2 feature map")

			plt.figure(figsize=(12,8))
			for i in range(36):
			    plt.subplot(6, 6, i+1)
			    temp = arr_2[i]
			    temp = cv2.resize(temp, (200, 66), cv2.INTER_AREA)
			    plt.imshow(temp)
			    plt.axis('off')
			plt.show()

			plt.figure(figsize=(12,8))
			for i in range(64):
			    plt.subplot(8, 8, i+1)
			    temp = arr_4[i]
			    temp = cv2.resize(temp, (200, 66), cv2.INTER_AREA)
			    plt.imshow(temp)
			    plt.axis('off')
			plt.show()

	plt.hist(angles, bins='auto')  # plt.hist passes it's arguments to np.histogram
	filename = 'output-%s.png' % datetime.now()
	plt.savefig(filename)
	print("Histogram saved to:", filename)

if __name__ == "__main__":
    main()