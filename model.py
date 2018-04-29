import os
import csv
import cv2
import numpy as np
import sklearn

samples = []

with open('TrainingData/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
	num_samples = len(samples)
	correction = 0.2
	#print(num_samples)
	while 1: # Loop forever so the generator never terminates
        	# sklearn.utils.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			#print("first for loop")
			batch_samples = samples[offset:offset+batch_size]
			images = []
			angles = []
			for batch_sample in batch_samples:
				#print("second for loop")
				for i in range(3):
					name = 'TrainingData/IMG/'+batch_sample[i].split('/')[-1]
					image = cv2.imread(name)
					angle = float(batch_sample[3])
					images.append(image)
					if i == 1:
						angle = angle + correction
					elif i == 2:
						angle = angle - correction
					angles.append(angle)

			# trim image to only see section with road
			#print("np array time")
			X_train = np.array(images)
			y_train = np.array(angles)
			#print("time to yield")
			yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
ch, row, col = 3, 160, 320  # Trimmed image format

#images = []
#measurements = []
#for line in lines:
#	for i in range(3):
#		source_path = line[i]
#		filename = source_path.split('/')[-1]
#		#print(filename)
#		current_path  = 'TrainingData/IMG/' + filename
#		image = cv2.imread(current_path)
#		images.append(image)
#		measurement = float(line[3])
#		correction = 0.2
#		if i == 1:
#			measurement = measurement + correction
#		elif i == 2:
#			measurement = measurement - correction 
#		measurements.append(measurement)
#
#		# augment - flip
#		image_flipped = np.fliplr(image)
#		measurement_flipped = -1.0*measurement
#		images.append(image_flipped)
#		measurements.append(measurement)
#
#X_train = np.array(images)
#y_train = np.array(measurements)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda 
from keras.layers.convolutional  import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Cropping2D



model = Sequential()
#model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(ch, row, col), output_shape=(ch, row, col)))
model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#model.fit_generator(train_generator, steps_per_epoch= len(train_samples), validation_data=validation_generator, validation_steps=len(validation_samples), epochs=5, verbose = 1)
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)
#history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('modelNVIDIA.h5')
#print(history_object.history.keys())
