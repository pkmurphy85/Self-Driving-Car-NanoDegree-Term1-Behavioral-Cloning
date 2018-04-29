import os
import csv
import cv2
import numpy as np
import sklearn

# Load training images
samples = []
with open('TrainingData/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

# Split training images into training and validation sets
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# generator function to process data in batches rather than loading all images into memmory
def generator(samples, batch_size=32):
	num_samples = len(samples)

	# correction factor for left and right camera images
	correction = 0.2
	while 1: # Loop forever so the generator never terminates
		# shuffle
		sklearn.utils.shuffle(samples)
		# batch process
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			images = []
			angles = []
			for batch_sample in batch_samples:
				# process center, left, and right images
				for i in range(3):
					name = 'TrainingData/IMG/'+batch_sample[i].split('/')[-1]
					image = cv2.imread(name)
					angle = float(batch_sample[3])
					images.append(image)

					# if left or right image, include correction factor
					if i == 1:
						angle = angle + correction
					elif i == 2:
						angle = angle - correction
					angles.append(angle)

			X_train = np.array(images)
			y_train = np.array(angles)

			# return batch
			yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional  import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Cropping2D

model = Sequential()
# Normalize
model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(160, 320, 3)))
# Crop out extraneous image data
model.add(Cropping2D(cropping=((70,25), (0,0))))
# Convolutions with relu activations
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
# Fully connected layers
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# MSE and adam optimizer
model.compile(loss='mse', optimizer='adam')

# Train model, 3 epochs
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)
# Save model
model.save('model.h5')
