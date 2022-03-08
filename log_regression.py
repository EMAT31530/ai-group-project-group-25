import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt


def produce_predictions_linear_regression(train_X, test_X, train_y):
    # function to find predictions using LinearRegression model from sklearn

    model = LinearRegression()
    model.fit(train_X, train_y)
    predictions = model.predict(test_X)

    return predictions


fs = pd.read_csv('FSW csv.csv')
# fs = pd.read_csv('FSW no rank.csv')
# fs = pd.read_csv('Feature Set Whole')
fs = fs.drop('Unnamed: 0', 1)

fs = fs.drop('8', 1)

remove = ['9']

labels = fs['9']
print(fs.columns)
features = fs.drop('9', 1)

for i in features.columns:
    features[i] = features[i].fillna(0)

min_max_scaler = preprocessing.MinMaxScaler()
features = min_max_scaler.fit_transform(features)

# for i in range(len(features)):
#     print(features.loc[[i]])

## Linear Regression
# okay = 0
# okay_set = []
# for i in range(0, 100):

    # train_X, test_X, train_y, test_y = train_test_split(features, labels, test_size=0.25, random_state=i)
    # # test_y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # predictions = produce_predictions_linear_regression(train_X, test_X, train_y)
    #
    # n = 0
    # for n in range(len(predictions)):
    #     rand = np.random.randint(0, 10000000)
    #     rad = rand/10000000
    #     if rad > predictions[n]:
    #         predictions[n] = 0
    #     else:
    #         predictions[n] = 1
    # # print(predictions)
    # act = np.array(test_y)
    # paa = []
    # sc = 0
    # for p in range(len(act)):
    #     pva = predictions[p], act[p]
    #     paa = paa + [pva]
    #     if predictions[p] == act[p]:
    #         sc += 1
    # perc_sc = sc*100/250
    # if perc_sc > 60:
    #     okay += 1
    #     okay_set += [perc_sc]
    # if score > 0.6:
    #     okay_set = okay_set + [score]

# Logistic Regression
clf = LogisticRegression(penalty='l1', solver='liblinear', max_iter=100).fit(features, labels)
pred = clf.predict(features)
prob = clf.predict_proba(features)
score = clf.score(features, labels)
intercept = clf.intercept_
coeffs = clf.coef_

prob = clf.predict_proba(features)
max_set = []
num = 0
inc = 0
cor = 0

for i in range(len(prob)):
    feats = (features[i])
    pred_proba = (prob[i])
    label = (labels[i])
    certainty = max(pred_proba)
    ind = np.where(pred_proba == certainty)[0]
    ind = ind[0]

    if certainty > 0.5:
        num += 1
        # print(i, label)
        # print(ind)
        if ind == label:
            cor += 1
        else:
            inc += 1
    max_set = max_set + [certainty]


# # plot histogram of certainties
# plt.hist(max_set, bins=20)
# plt.show()

# model = LinearRegression().fit(train_X, train_y)

# print(okay_set)
# print(len(okay_set))
print(num)
print(cor)
print(coeffs)
