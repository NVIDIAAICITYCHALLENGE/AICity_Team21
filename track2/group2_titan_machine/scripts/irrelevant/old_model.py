import pickle
import os
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import load_model, Model
from keras.layers import Input, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.applications.vgg16 import VGG16


def fetch_data(dataset):

	def batcherator(chunk_size):
		chunk_start = 0
		with h5py.File('/home/aicg2/data/data.h5', 'r') as f:
			x_train, x_test = np.array([1]), np.array([1])
			while x_train.size or x_test.size:
				x_train = f[dataset]['train'][chunk_start:chunk_start + chunk_size]
				x_test = f[dataset]['val'][chunk_start:chunk_start + chunk_size] 
				chunk_start += chunk_size
				yield x_train, x_test	

	scaler = StandardScaler()

	x_train = x_test = None
	for train_chunk, test_chunk in batcherator(1000):
		print "train_chunk and test_chunk shape:", train_chunk.shape, test_chunk.shape
		if train_chunk.shape[0]:
			scaler.partial_fit(train_chunk.reshape(train_chunk.shape[0], -1))
		if test_chunk.shape[0]:
			scaler.partial_fit(test_chunk.reshape(test_chunk.shape[0], -1))	

		if x_train is not None and x_test is not None:
			x_train = np.vstack((x_train, train_chunk))
			x_test = np.vstack((x_test, test_chunk))
			print "x_train and x_test shape:", x_train.shape, x_test.shape
		else:
			x_train = train_chunk
			x_test = test_chunk
	
	print "Beginning data transformation..."

	start, size = 0, 1000
	while start < x_train.shape[0]:
		print "Transforming train chunk starting at", start
		data = x_train[start:start + size]
		x_train[start:start + size] = scaler.transform(
			data.reshape(data.shape[0], -1)
		).reshape(data.shape)
		start += size

	start, size = 0, 1000
	while start < x_test.shape[0]:
		print "Transforming test chunk starting at", start
		data = x_test[start:start + size]
		x_test[start:start + size] = scaler.transform(
			data.reshape(data.shape[0], -1)
		).reshape(data.shape)
		start += size

	train_labels = '/home/aicg2/group2/scripts/single_class_labels/{}_train_labels.txt'.format(dataset)
	test_labels = '/home/aicg2/group2/scripts/single_class_labels/{}_val_labels.txt'.format(dataset)

	y_train = np.genfromtxt(train_labels, dtype=int)
	y_test = np.genfromtxt(test_labels, dtype=int)

	return x_train, x_test, y_train, y_test


def initialize_model(image_shape, train_conv_layers=False):

	# Custom input layer
	input = Input(shape=image_shape, name='image_input')

	# Load convolutional block layers of the VGG16 model
	initial_model = VGG16(weights='imagenet', include_top=False)

	# If not `train_conv_layers`, do not train VGG16 convolutional blocks 
	if not train_conv_layers:
		for layer in initial_model.layers:
			layer.trainable = False
			
	# Add top layers to combine features and predict continuous values
	x = Flatten()(initial_model(input))
	x = Dense(300, activation='relu')(x)
	x = BatchNormalization()(x)
	x = Dropout(0.5)(x)
	x = Dense(500, activation='relu')(x)
	x = BatchNormalization()(x)
	x = Dropout(0.5)(x)
	x = Dense(1, activation='linear')(x)

	# Make new model and compile it
	model = Model(inputs=input, outputs=x)	
	model.compile(loss='mse', optimizer='adam', metrics=['mae'])

	print(model.summary())

	return model


def train_model(epochs=10, batch_size=8):
	model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
	score = model.evaluate(x_test, y_test, batch_size=batch_size)

	print "Evaluation score: {}".format(score)
	model.save('model.h5')


if __name__ == '__main__':
	x_train, x_test, y_train, y_test = fetch_data(dataset='aic480')
	
	if not os.path.isfile('model.h5'):
		model = initialize_model(x_train.shape[0], train_conv_layers=False)
	else:
		model = load_model('model.h5')

	train_model(epochs=10)
