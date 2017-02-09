import matplotlib.pyplot as plt
from generators import random_shear, crop, resize, random_shadows, random_flip, random_brightness

img = plt.imread("img/pipeline.jpg")
plt.imshow(img)
plt.show()
resize_dim= (64,64)
steering_angle = 0
image, steering_angle = random_shear(img, steering_angle)

print("After shear")
print(steering_angle)
plt.imshow(image)
plt.show()

image = crop(image)

print("After crop")
plt.imshow(image)
plt.show()

image = resize(image, resize_dim)
print("After resize")
plt.imshow(image)
plt.show()

image = random_shadows(image)
print("After shadows")
plt.imshow(image)
plt.show()

image, steering_angle = random_flip(image, steering_angle)
print("After flip")
print(steering_angle)
plt.imshow(image)
plt.show()

image = random_brightness(image)
print("After brightness")
plt.imshow(image)
plt.show()

