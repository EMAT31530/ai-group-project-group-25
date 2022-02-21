#import relevant modules
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow
import pandas as pd

matches = pd.read_csv('allmatches.csv')
features_list = matches.columns

#Split into training and test data first
results = matches['Winner'] #outcomes of matches in the data
features = features_list #column headings list
num_features = len(features)
num_classes = np.unique(results)
print(np.shape(matches))

features_train, features_test, results_train, results_test = train_test_split(features, results, test_size = 0.15) #can add random_state = int to shuffle the data

scaler = sklearn.preprocessing.StandardScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

#Use 'relu' or 'sigmoid' for activation
#Use 'softmax' or 'sigmoid' for output layer
#Initialise neural network
model = tensorflow.keras.models.Sequential()
model.add(tensorflow.keras.layers.Dense(32,input_dim=num_features,activation = 'relu'))
model.add(tensorflow.keras.layers.Dense(16,activation='relu'))
model.add(tensorflow.keras.layers.Dense(8,activation='relu'))
model.add(tensorflow.keras.layers.Dense(4,activation='relu'))
model.add(tensorflow.keras.layers.Dense(num_classes,activation='softmax'))

#calculate cross-entropy loss
model.compile(loss='categorical_crossentropy', optimizer=tensorflow.keras.optimisers.SGD(learning_rate=0.01), metrics=['accuracy'])

history = model.fit(features_train,results_train, epochs=500)
