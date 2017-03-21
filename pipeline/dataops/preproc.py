#!/usr/bin/env/python3.5

"""
Functions for data preprocessing. 
"""

import _pickle as cPickle
import numpy as np

def add_gnoise(x_train, mu=0, sigma=1):
    """add_gnoise( x_train, mu=0, sigma=1 )
    """
    train_noise = np.random.normal(
    loc=mu, scale=sigma, size=x_train.shape) / 255
    x_train += train_noise
    return np.clip(x_train, a_min=0, a_max=1, out=None)

def save_gnoisy(x_train, gauss_params):
    """save_gnoisy(x_train, gauss_params)
    """
    for (mu, sigma) in gauss_params:
        x_train = add_gnoise(x_train, mu, sigma)
        with open("./data/gnoisy/x_train_{}n{}.p".format(mu, sigma), "wb") as outfile:
            cPickle.dump(x_train, outfile)

def corrupt_labels(y_train, percent_lerr):
    """ corrupt_labels(y_train, percent_lerr)
	(int ndarray) y_train is binary valued numpy array n_train x n_labels
    """
    nlabels, nclasses = y_train.shape[0], y_train.shape[1]
    y_train_labels = np.argmax(y_train, axis=1) # create a list of labels 0-9 of length n_train
    nlabelerr = min(nlabels, int(np.round(nlabels*percent_lerr))) # min to avoid overflow. round rounds 0.5 to 0.0  
    err_indices = np.random.permutation(np.arange(y_train.shape[0]))[0:nlabelerr] 
    for i in err_indices:
        mislabel = y_train_labels[i]
        while mislabel == y_train_labels[i]:
            mislabel = np.random.randint(low = 0, high=nclasses, size=None, dtype='l') 
        y_train[i][:] = np.zeros(nclasses)
        y_train[i][mislabel] = 1
    return y_train

def save_corrupted(y_train, percent_lerr_tuple):
    """
    """
    for percent_lerr in percent_lerr_tuple:
        y_train_lerr = corrupt_labels(y_train, percent_lerr)
        with open( "./data/lerr/y_train_lerr_{}per.p".format(percent_lerr), "wb" ) as outfile:
            cPickle.dump( y_train_lerr, outfile )

