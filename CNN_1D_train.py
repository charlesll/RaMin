# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 14:29:39 2017

@author: charles
"""

import os

import pickle
import numpy as np

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
import keras

#
# Data importation
#
print('Loading data...')
X = np.load('./data/obs.npy')
X_sc = StandardScaler().fit(X) # standard scaling
X = X_sc.transform(X)
y = pickle.load( open( "./data/labels.pkl", "rb" ) )

X_i, X_test, y_i, y_test = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True)
X_train, X_valid, y_train, y_valid = train_test_split(X_i, y_i, test_size=0.20, random_state=42, shuffle=True)

print(X_train.shape)

# Reshaping
X_train = np.expand_dims(X_train, axis=2)
X_valid = np.expand_dims(X_valid, axis=2)
X_test = np.expand_dims(X_test, axis=2)

print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

#print(y_train)
nb_class = 27
y_train = keras.utils.to_categorical(y_train, nb_class)
y_test = keras.utils.to_categorical(y_test, nb_class)
y_valid = keras.utils.to_categorical(y_valid, nb_class)

#%%
print('Build model...')
model = Sequential()

# create the model
model.add(Conv1D(4,2,activation='relu',kernel_initializer='glorot_uniform',input_shape=(600,1)))
model.add(MaxPooling1D())
model.add(Dropout(0.25))

#model.add(Conv1D(2,2,activation='tanh',kernel_initializer='glorot_uniform'))
#model.add(MaxPooling1D())
#model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(100, activation='relu'))

model.add(Dense(nb_class, activation='softmax'))

early_stopping = EarlyStopping(monitor='val_loss', patience=0)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=100, batch_size=8,callbacks=[early_stopping])

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
