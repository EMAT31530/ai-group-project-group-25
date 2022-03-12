# import relevant modules
from sklearn.model_selection import train_test_split
import sklearn
import time
import numpy as np
import tensorflow
import pandas as pd
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model

from numpy.random import seed
from tensorflow import random

start_time = time.time()

seed(0)
random.set_seed(0)
import os

os.environ['TF_DETERMINISTIC_OPS'] = '1'

def neural_network(data):

    matches = pd.read_csv('aus_diff_features.csv')
    matches = matches.iloc[0:28960, :]
    matches = matches.dropna()
    x = matches.outcome  # outcomes of matches in the data
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
    d = sklearn.preprocessing.normalize(y, axis=1)
    new_y = pd.DataFrame(d, columns=features_list)

    num_classes = len(np.unique(x))
    print(np.unique(x))
    num_features = len(features_list)

    # Split into training and test data first
    y_train, y_test, x_train_categorical, x_test_categorical = train_test_split(new_y, x, test_size=0.2,
                                                                                random_state=2)  # can add random_state = int to shuffle the data

    # scaler = sklearn.preprocessing.StandardScaler()
    # y_train = scaler.fit_transform(y_train)
    # y_test = scaler.transform(y_test)

    # Use 'relu' or 'sigmoid' for activation
    # Use 'softmax' or 'sigmoid' for output layer
    # Initialise neural network
    model = tensorflow.keras.models.Sequential()
    model.add(tensorflow.keras.layers.Dense(32, input_dim=num_features, activation='relu'))
    model.add(tensorflow.keras.layers.Dense(16, activation='relu'))
    model.add(tensorflow.keras.layers.Dense(num_classes, activation='softmax'))

    # calculate cross-entropy loss
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.000001), metrics=['accuracy'])

    model.summary()

    # checkpoint_path="weights.{epoch:02d}-{val_loss:.2f}.h5"
    modelcheckpoint = tensorflow.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True,
                                                                 mode='min', verbose=2)

    # Train the model with the new callback
    history = model.fit(y_train, x_train_categorical, batch_size=128, validation_data=(y_test, x_test_categorical),
                        epochs=2000, callbacks=[modelcheckpoint])

    #model.save("model.h5")

    best_model = load_model('best_model.h5')

    # evaluate the model
    scores = model.evaluate(y_train, x_train_categorical, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    # review results
    train_acc = best_model.evaluate(y_train, x_train_categorical, verbose=0)
    test_acc = best_model.evaluate(y_test, x_test_categorical, verbose=0)
    print(train_acc)
    print(test_acc)

    # for predictions
    # put matches throught create_features call it ausopen22.csv (for example)
    ausopen22 = data
    aus_matches = ausopen22[features_list]

    # add new column for predictions
    predictions = best_model.predict(aus_matches)
    actual_pred = []
    for pred in predictions:
        preds = np.argmax(pred)
        actual_pred.append(preds)

    #new_pred = []
    #for i in actual_pred:
    #    if i == 0:
    #        new_pred.append(1)
    #    else:
    #        new_pred.append(0)
    #print(new_pred)
    ausopen22 = ausopen22.assign(predictions=actual_pred)

    # add new column for probabilities
    probabilities = best_model.predict(aus_matches)
    print(probabilities)
    df = pd.DataFrame(probabilities, columns=["outcome 0 probability", "outcome 1 probability"])

    aus22_output = ausopen22[['player_0', 'player_1', 'predictions']]
    indices = np.arange(0, len(aus22_output))
    aus22_output = aus22_output.set_index(indices)

    finaldf = pd.concat([aus22_output, df], axis=1)
    print(finaldf)

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

    return finaldf, compare


new_matches = pd.read_csv('aus_diff_features.csv')
first_round = new_matches.iloc[28960:29024, :]
second_round = new_matches.iloc[29024:29054, :]
third_round = new_matches.iloc[29054:29070,:]
fourth_round = new_matches.iloc[29070:29078,:]
quarter_finals = new_matches.iloc[29078:29082,:]
semi_finals = new_matches.iloc[29082:29084,:]
final = new_matches.iloc[29084,:]
finaldf, results = neural_network(first_round)
print(finaldf)
print(results)

player_0_wins = results[results['outcome'] == 1].reset_index(drop=True)
player_1_wins = results[results['outcome'] == 0].reset_index(drop=True)

for i in range(player_0_wins.shape[0]):

    verdict_0 = player_0_wins['verdict']
    predictions_0 = player_0_wins['predictions']
    player_0_0 = player_0_wins['player_0']
    player_1_0 = player_0_wins['player_1']

    if verdict_0[i] == 'Wrong':

        if second_round['player_0'].str.contains(player_1_0[i]).any():
            second_round['player_0'] = second_round['player_0'].replace({player_1_0[i]:player_0_0[i]})
        if second_round['player_1'].str.contains(player_1_0[i]).any():
            second_round['player_1'] = second_round['player_1'].replace({player_1_0[i]:player_0_0[i]})

for i in range(player_1_wins.shape[0]):

    verdict_1 = player_1_wins['verdict']
    predictions_1 = player_1_wins['predictions']
    player_0_1 = player_1_wins['player_0']
    player_1_1 = player_1_wins['player_1']

    if verdict_1[i] == 'Wrong':

        if second_round['player_0'].str.contains(player_0_1[i]).any():
            second_round['player_0'] = second_round['player_0'].replace({player_0_1[i]: player_1_1[i]})
        if second_round['player_1'].str.contains(player_0_1[i]).any():
            second_round['player_1'] = second_round['player_1'].replace({player_0_1[i]: player_1_1[i]})

# second_results = neural_network(second_round)

print("--- %s seconds ---" % (time.time() - start_time))




