#import relevant modules
from sklearn.model_selection import train_test_split
import sklearn
import numpy as np
import tensorflow
import pandas as pd
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

from numpy.random import seed
from tensorflow import random
seed(0)
random.set_seed(0)
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'

matches = pd.read_csv('diff_features.csv')
x = matches.Winner #outcomes of matches in the data
label_encoder = LabelEncoder()
x = label_encoder.fit_transform(x)
x=to_categorical(x)

features_list = [ 'WRank', 'LRank', 'WPts', 'LPts', 'W1', 'L1', 'W2', 'L2', 'W3', 'L3', 'W4', 'L4', 'W5', 'L5', 'Wsets', 'Lsets',] #change this - don't use all columns
y = matches[features_list]

num_classes = len(np.unique(y))
num_features = len(features_list)

#Split into training and test data first
y_train, y_test, x_train_categorical, x_test_categorical = train_test_split(y, x, test_size = 0.15) #can add random_state = int to shuffle the data #can add random_state = int to shuffle the data

scaler = sklearn.preprocessing.StandardScaler()
y_train = scaler.fit_transform(y_train)
y_test = scaler.transform(y_test)

#Use 'relu' or 'sigmoid' for activation
#Use 'softmax' or 'sigmoid' for output layer
#Initialise neural network
model = tensorflow.keras.models.Sequential()
model.add(tensorflow.keras.layers.Dense(32,input_dim=num_features,activation = 'relu'))
model.add(tensorflow.keras.layers.Dense(16,activation='relu'))
model.add(tensorflow.keras.layers.Dense(8,activation='relu'))
model.add(tensorflow.keras.layers.Dense(4,activation='relu'))
model.add(tensorflow.keras.layers.Dense(591,activation='softmax'))

#calculate cross-entropy loss
model.compile(loss='categorical_crossentropy', optimizer=tensorflow.keras.optimisers.SGD(learning_rate=0.01), metrics=['accuracy'])

model.summary()

history = model.fit(features_train,results_train, epochs=500)

#review results
train_acc = model.evaluate(y_train, x_train_categorical, verbose=0)
test_acc = model.evaluate(y_test, x_test_categorical, verbose=0)
print(train_acc)
print(test_acc)

model = tensorflow.keras.models.Sequential()
model.add(tensorflow.keras.layers.Dense(32,input_dim=num_features,activation = 'relu'))
model.add(tensorflow.keras.layers.Dense(16,activation='relu'))
model.add(tensorflow.keras.layers.Dense(8,activation='relu'))
model.add(tensorflow.keras.layers.Dense(4,activation='relu'))
model.add(tensorflow.keras.layers.Dense(591,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.01), metrics=['accuracy'])

checkpoint_path="weights.best.hdf5"
checkpoint = tensorflow.keras.callbacks.ModelCheckpoint(checkpoint_path,monitor='val accuracy', verbose=0, save_best_only=True, mode = 'max')

callbacks_list = [checkpoint]

# Train the model with the new callback
model.fit(y_train, x_train_categorical,epochs=100, callbacks=[callbacks_list],verbose=0)
