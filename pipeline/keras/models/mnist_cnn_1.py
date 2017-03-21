'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU. --- Edit this comment and add citation, ma
ke sure doc strings are in place.
'''
from __future__ import print_function

def run_cnn_model1(train, test, model_id=""):
	"""Returns (callbacks, model), where callbacks is a tuple of callbacks
	"""
	import os #avoid importing the entire module
	import keras #avoid importing the entire module
	from keras.models import Sequential, load_model
	from keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D

	## checkpointer callback imports ##
	###################################
	from keras.callbacks import ModelCheckpoint, TensorBoard
	from ..callbacks import LossHistory
	
	x_train = train[0]
	y_train = train[1]
	x_test = test[0]
	y_test = test[1]
	batch_size = 128
	num_classes = 10
	epochs = 1

	img_rows, img_cols = 28, 28
	input_shape = (img_rows, img_cols, 1)

	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),
			 input_shape=input_shape))
	convout1 = Activation('relu')
	model.add(convout1)
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(128))
	cnn_codes = Activation('relu')
	model.add(cnn_codes)
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	model.compile(loss=keras.losses.categorical_crossentropy,
		      optimizer=keras.optimizers.Adadelta(),
		      metrics=['accuracy'])

	# If model directory does not exist, make one
	model_name = "mnist_cnn_{}".format(model_id)
	model_dir = "./results/" + model_name
	checkfile = "/chckpt_weights.{epoch:02d}-{val_loss:.2f}.hdf5"
	filepath = model_dir + checkfile
	if not os.path.exists(model_dir):
	 		os.makedirs(model_dir)
	
	## Callbacks
	loss_history = LossHistory()	
	
	# Warning: don't use .format() method {} conflicts with keras parsing, tuple out of bounds error
	checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)
	history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
		  verbose=0, validation_data=(x_test, y_test), callbacks = [checkpointer, loss_history])
	
	# saves model as an HDF5 file
	model.save(model_dir+"/{}".format(model_name) + ".h5")

	score = model.evaluate(x_test, y_test, verbose=0)
	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	return ((loss_history, checkpointer, history), model)

