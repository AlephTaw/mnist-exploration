#!/usr/bin/env/ python3

"""
Functions for loading and formatting data.
"""

from keras import backend as K
from keras.datasets import mnist
import _pickle as cPickle
import numpy
import keras

	
def formatted_mnist(img_rows = 28, img_cols = 28, num_classes = 10):
	"""the data, shuffled and split between train and test sets
	"""
	
	num_classes = 10
	img_rows = 28
	img_cols = 28

	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	
	if K.image_data_format() == 'channels_first':
		x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
		x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
		input_shape = (1, img_rows, img_cols)
	else:
		x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
		x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
		input_shape = (img_rows, img_cols, 1)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255

	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)
	
	return (input_shape, (x_train, y_train), (x_test, y_test))

def load_lerr_data(percent_lerr):
    """
    """
    with open( "./data/lerr/y_train_lerr_{}per.p".format(percent_lerr), "rb" ) as infile:
            return cPickle.load(infile)

def load_gnoisy_data(mu, sigma):
    """
    """
    with open( "./data/gnoisy/x_train_{}n{}.p".format(mu, sigma), "rb" ) as infile:
            return cPickle.load(infile)
