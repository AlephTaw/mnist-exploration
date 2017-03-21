#!/usr/bin/env/ python3

"""
Custom Keras callbacks.
"""
import keras

## custom callback saves losses after each batch
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

