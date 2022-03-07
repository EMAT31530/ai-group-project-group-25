#import relevant modules
from sklearn.model_selection import train_test_split
import sklearn
import numpy as np
import tensorflow
import pandas as pd
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model

from numpy.random import seed
from tensorflow import random
seed(0)
random.set_seed(0)
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'

matches = pd.read_csv('aus_diff_features.csv')
matches = matches.iloc[0:28960,:]
matches = matches.dropna()
x = matches.outcome #outcomes of matches in the data
label_encoder = LabelEncoder()
x = label_encoder.fit_transform(x)
x = to_categorical(x)

features_list = ['diff_rank','diff_match_win_percentage','diff_match_win_percentage_hh','diff_match_win_percentage_sets','diff_match_win_percentage_surface','diff_match_win_percentage_surface_sets','diff_game_win_percentage','diff_game_win_percentage_hh','diff_game_win_percentage_sets','diff_game_win_percentage_surface','diff_game_win_percentage_surface_sets']
y = matches[features_list]

num_classes = len(np.unique(x))
num_features = len(features_list)

#Split into training and test data first
y_train, y_test, x_train_categorical, x_test_categorical = train_test_split(y, x, test_size = 0.2) #can add random_state = int to shuffle the data

scaler = sklearn.preprocessing.StandardScaler()
y_train = scaler.fit_transform(y_train)
y_test = scaler.transform(y_test)

#Use 'relu' or 'sigmoid' for activation
#Use 'softmax' or 'sigmoid' for output layer
#Initialise neural network
model = tensorflow.keras.models.Sequential()
model.add(tensorflow.keras.layers.Dense(32,input_dim = num_features, activation = 'relu'))
model.add(tensorflow.keras.layers.Dense(16,activation='relu'))
model.add(tensorflow.keras.layers.Dense(8,activation='relu'))
model.add(tensorflow.keras.layers.Dense(4,activation='relu'))
model.add(tensorflow.keras.layers.Dense(2,activation='sigmoid'))

#calculate cross-entropy loss
model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.01), metrics=['accuracy'])

model.summary()

#checkpoint_path="weights.{epoch:02d}-{val_loss:.2f}.h5"
modelcheckpoint = tensorflow.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode = 'min', verbose = 2)

# Train the model with the new callback
history=model.fit(y_train, x_train_categorical, batch_size=128, validation_data=(y_test,x_test_categorical),epochs=900,callbacks=[modelcheckpoint], validation_split=0.1)

# evaluate the model
scores = model.evaluate(y_train, x_train_categorical, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#review results
train_acc = model.evaluate(y_train, x_train_categorical, verbose=0)
test_acc = model.evaluate(y_test, x_test_categorical, verbose=0)
print(train_acc)
print(test_acc)

# save model and architecture to single file
#model.save("model.h5")

best_model = load_model('best_model.h5')

#for predictions
#put matches throught create_features call it ausopen22.csv (for example)
new_matches = pd.read_csv('aus_diff_features.csv')
ausopen22 = new_matches.iloc[28960:29024,:]
aus_matches = ausopen22[features_list]

#add new column for predictions
predictions = best_model.predict(aus_matches)
actual_pred = []
for pred in predictions:
    preds = np.argmax(pred)
    actual_pred.append(preds)

ausopen22 = ausopen22.assign(predictions = actual_pred)

#add new column for probabilities
probabilities = best_model.predict(aus_matches)
print(probabilities)
df = pd.DataFrame(probabilities, columns=["outcome 0 probability","outcome 1 probability"])

aus22_output = ausopen22[['player_0','player_1','predictions']]
indices = np.arange(0,len(aus22_output))
aus22_output = aus22_output.set_index(indices)

finaldf = pd.concat([aus22_output,df],axis = 1)
print(finaldf)