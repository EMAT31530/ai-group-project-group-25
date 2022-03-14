#import relevant modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import sklearn
import time
import numpy as np
import tensorflow
import pandas as pd
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
import matplotlib.pyplot as plt

from numpy.random import seed
from tensorflow import random


start_time = time.time()

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

features_list = ['diff_rank', 'diff_match_win_percentage_year', 'diff_match_win_percentage',
                     'diff_match_win_percentage_hh', 'diff_match_win_percentage_sets',
                     'diff_match_win_percentage_surface', 'diff_match_win_percentage_year_surface',
                     'diff_match_win_percentage_surface_sets', 'diff_game_win_percentage_year',
                     'diff_game_win_percentage', 'diff_game_win_percentage_hh',
                     'diff_game_win_percentage_sets', 'diff_game_win_percentage_surface',
                     'diff_game_win_percentage_year_surface', 'diff_game_win_percentage_surface_sets']

y = matches[features_list]
#d = sklearn.preprocessing.normalize(y,axis=1)
#new_y = pd.DataFrame(d,columns=features_list)

num_classes = len(np.unique(x))
num_features = len(features_list)

#Split into training and test data first
y_train, y_test, x_train_categorical, x_test_categorical = train_test_split(y, x, test_size = 0.2, random_state = 2)

#scaler = sklearn.preprocessing.StandardScaler()
#y_train = scaler.fit_transform(y_train)
#y_test = scaler.transform(y_test)

#Use 'relu' or 'sigmoid' for activation
#Use 'softmax' or 'sigmoid' for output layer
#Initialise neural network
model = tensorflow.keras.models.Sequential()
model.add(tensorflow.keras.layers.Dense(64,input_dim = num_features, activation = 'relu'))
model.add(tensorflow.keras.layers.Dense(16,activation='relu'))
#model.add(tensorflow.keras.layers.Dense(8,activation='relu'))
model.add(tensorflow.keras.layers.Dense(num_classes,activation='softmax'))

#calculate cross-entropy loss
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

model.summary()

#checkpoint_path="weights.{epoch:02d}-{val_loss:.2f}.h5"
modelcheckpoint = tensorflow.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode = 'min', verbose = 2)

# Train the model with the new callback
history=model.fit(y_train, x_train_categorical, batch_size=128, validation_data=(y_test,x_test_categorical),epochs=2000,callbacks=[modelcheckpoint])

# save model and architecture to single file
#model.save("best_model.h5")

best_model = load_model('best_model.h5')

# evaluate the model
scores = best_model.evaluate(y_train, x_train_categorical, verbose=0)
print("%s: %.2f%%" % (best_model.metrics_names[1], scores[1]*100))

#review results
train_acc = best_model.evaluate(y_train, x_train_categorical, verbose=0)
test_acc = best_model.evaluate(y_test, x_test_categorical, verbose=0)
print(train_acc)
print(test_acc)

#for predictions

new_matches = pd.read_csv('aus_diff_features.csv')
ausopen22 = new_matches.iloc[28960:,:]
aus_matches = ausopen22[features_list]

#add new column for predictions
predictions = best_model.predict(aus_matches)
actual_pred = []
for pred in predictions:
    preds = np.argmax(pred)
    actual_pred.append(preds)

ausopen22 = ausopen22.assign(predictions=actual_pred)

#add new column for probabilities
probabilities = best_model.predict(aus_matches)
df = pd.DataFrame(probabilities, columns=["outcome 0 probability","outcome 1 probability"])

aus22_output = ausopen22[['player_0','player_1','predictions']]
indices = np.arange(0,len(aus22_output))
aus22_output = aus22_output.set_index(indices)

finaldf = pd.concat([aus22_output,df],axis = 1)
#print(finaldf)
print(finaldf[['predictions','outcome 0 probability','outcome 1 probability']])

actual_results = ausopen22['outcome'].reset_index(drop=True)
predicted_results = finaldf[['player_0', 'player_1', 'predictions']]

compare = pd.concat([predicted_results, actual_results], axis = 1)

verdict = []
correct = 0
wrong = 0

for i in range(compare.shape[0]):

    if actual_results[i] == predicted_results.iloc[i, 2]:
        verdict.append('Correct')
        correct += 1
    else:
        verdict.append('Wrong')
        wrong += 1
compare['verdict'] = verdict
accuracy = correct / (correct + wrong)
print(accuracy)

#accuracy graphs over time
#plots at 20 epoch intervals
plt.title('How the accuracy varies with Epochs')
plt.plot(history.epoch[::20], history.history['accuracy'][::20], label='Train')
plt.plot(history.epoch[::20], history.history['val_accuracy'][::20], label='Test')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'])
plt.show()

#loss graphs over time
#plots at 20 epoch intervals
plt.title('How the loss varies with Epochs')
plt.plot(history.epoch[::20], history.history['loss'][::20], label='Train')
plt.plot(history.epoch[::20], history.history['val_loss'][::20], label='Test')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'])
plt.show()
#print(best_model.get_weights())
