import argparse
import numpy as np
import os
import csv
from scipy import ndimage
from keras.models import model_from_json

from PIL import Image

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Remote Driving')
	parser.add_argument(
		'model', type=str, help='Path to model definition json. Model weights should be on the same path.')
	parser.add_argument(
		'folder', type=str, help='Path to the data directory containing the driving log and the csv files.')
	args = parser.parse_args()
	with open(args.model, 'r') as jfile:
		# NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
		# then you will have to call:
		#
		#   model = model_from_json(json.loads(jfile.read()))\
		#
		# instead.
		model = model_from_json(jfile.read())

	basedir = args.folder
	log_file = os.path.join(basedir, 'driving_log.csv')

	with open(log_file, 'r') as csvfile:
		reader = csv.reader(csvfile)
		data = np.array(list(reader))
	
	X_input = data[:, 0]
	y_input = np.array(data[:, 3], dtype="float32")

	model.compile("adam", "mse")
	weights_file = args.model.replace('json', 'h5')
	model.load_weights(weights_file)

	imgs = [0, 55, 108]
	X_test = np.array([ndimage.imread(X_input[i]) for i in imgs])
	y_test = np.array([y_input[i] for i in imgs])
	# img = Image.fromarray(X_test[0])
	# img.show()
	# img = Image.fromarray(X_test[1])
	# img.show()
	# img = Image.fromarray(X_test[2])
	# img.show()

	# A zero steering sample
	# loss_and_metrics = model.evaluate(X_test, y_test, batch_size=32)
	predictions = model.predict(X_test, verbose=1)
	print("img labels %s" % y_test)
	print("predictions %s" % predictions)