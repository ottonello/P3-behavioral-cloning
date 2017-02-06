from generators import generate_next_batch
from PIL import Image
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH='./udacity'
DRIVING_LOG_FILE = DATA_PATH + '/driving_log.csv'

def main():
	csv = pd.read_csv(DRIVING_LOG_FILE)
	
	gen = generate_next_batch(csv, batch_size=10)
	imgs, angles = next(gen)

	show_images = True
	if show_images:
		for img, angle in zip(imgs,angles):
			print(angle)
			Image.fromarray(img).show(title = "test")
			input("Press Enter to continue...")

	plt.hist(angles, bins='auto')  # plt.hist passes it's arguments to np.histogram
	filename = 'output-%s.png' % datetime.now()
	plt.savefig(filename)
	print("Histogram saved to:", filename)

if __name__ == "__main__":
    main()