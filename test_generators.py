import numpy as np
import matplotlib.pyplot as plt
from generators import generate_next_batch
from PIL import Image

def main():
	gen = generate_next_batch(batch_size=10)
	imgs, angles = next(gen)

	show_images = True
	if show_images:
		for img, angle in zip(imgs,angles):
			print(angle)
			Image.fromarray(img).show(title = "test")
			input("Press Enter to continue...")

	plt.hist(angles, bins='auto')  # plt.hist passes it's arguments to np.histogram
	plt.show()

if __name__ == "__main__":
    main()